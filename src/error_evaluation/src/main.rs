use anyhow::{bail, Context, Result};
use bio::alignment::pairwise::{Aligner, MatchParams};
use bio::alignment::AlignmentOperation;
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

#[derive(Parser, Debug)]
#[command(name = "incremental_repeat_error")]
#[command(about = "Compute incremental (k=1..K) repeat-consensus error from split/bed/reference files.")]
struct Args {
    /// Input split file pattern, e.g. "/path/*_split.fa"
    #[arg(long = "split-glob", default_value = "*_split.fa")]
    split_glob: String,

    /// Maximum repeat count k (reports k=1..K)
    #[arg(long = "max-k", default_value_t = 4)]
    max_k: usize,

    /// Optional include filter applied to sample stem basename
    #[arg(long = "include-sample-substring", default_value = "")]
    include_sample_substring: String,

    /// Optional exclude filter applied to sample stem basename
    #[arg(long = "exclude-sample-substring", default_value = "")]
    exclude_sample_substring: String,

    /// Output TSV path
    #[arg(short = 'o', long = "output-tsv", default_value = "incremental_repeat_error.tsv")]
    output_tsv: PathBuf,

    /// Optional output TSV path for per-transition error rates by k
    #[arg(long = "transition-output-tsv", default_value = "")]
    transition_output_tsv: String,

    /// Number of worker threads (0 = auto)
    #[arg(long = "jobs", default_value_t = 0)]
    jobs: usize,

    /// Resume from existing output TSV by skipping completed samples
    #[arg(long = "resume", default_value_t = false)]
    resume: bool,
}

#[derive(Clone, Debug)]
struct SampleStats {
    callable: Vec<u64>,
    mismatch: Vec<u64>,
    callable_by_ref: Vec<[u64; 4]>,
    mismatch_by_sub: Vec<[u64; 12]>,
}

impl SampleStats {
    fn new(max_k: usize) -> Self {
        Self {
            callable: vec![0; max_k + 1],
            mismatch: vec![0; max_k + 1],
            callable_by_ref: vec![[0; 4]; max_k + 1],
            mismatch_by_sub: vec![[0; 12]; max_k + 1],
        }
    }
}

const SUBSTITUTIONS: [&str; 12] = [
    "A>C", "A>G", "A>T", "C>A", "C>G", "C>T", "G>A", "G>C", "G>T", "T>A", "T>C", "T>G",
];

fn sub_idx(ref_i: usize, alt_i: usize) -> Option<usize> {
    match (ref_i, alt_i) {
        (0, 1) => Some(0),
        (0, 2) => Some(1),
        (0, 3) => Some(2),
        (1, 0) => Some(3),
        (1, 2) => Some(4),
        (1, 3) => Some(5),
        (2, 0) => Some(6),
        (2, 1) => Some(7),
        (2, 3) => Some(8),
        (3, 0) => Some(9),
        (3, 1) => Some(10),
        (3, 2) => Some(11),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct Coord {
    start_1based: usize,
}

fn split_key_from_header(header: &str) -> String {
    let h = header.strip_prefix('@').unwrap_or(header);
    h.split('_').next().unwrap_or(h).to_string()
}

fn region_key_from_header(header: &str) -> String {
    let h = header.strip_prefix('>').unwrap_or(header);
    h.split("::").next().unwrap_or(h).to_string()
}

fn phred(c: u8) -> i32 {
    c as i32 - 33
}

fn base_idx(b: u8) -> Option<usize> {
    match b.to_ascii_uppercase() {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

fn consensus_base(votes: &[u8]) -> u8 {
    if votes.is_empty() {
        return b'N';
    }
    let mut c = [0usize; 4];
    for &b in votes {
        if let Some(i) = base_idx(b) {
            c[i] += 1;
        }
    }
    let pairs = [(b'A', c[0]), (b'C', c[1]), (b'G', c[2]), (b'T', c[3])];
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    if sorted[0].1 == 0 {
        return b'N';
    }
    if sorted.len() > 1 && sorted[0].1 == sorted[1].1 {
        return b'N';
    }
    // Unanimous-only consensus: any disagreement is ambiguous.
    if sorted.len() > 1 && sorted[1].1 > 0 {
        return b'N';
    }
    sorted[0].0
}

fn rand_aln_p(bl: usize, sl: usize) -> f64 {
    if sl < bl {
        1.0 - (1.0 - 0.25_f64.powi(sl as i32)).powi((bl.saturating_sub(sl)) as i32)
    } else {
        1.0 - (1.0 - 0.25_f64.powi(bl as i32)).powi((sl.saturating_sub(bl)) as i32)
    }
}

fn load_bed_first_coord(path: &PathBuf) -> Result<HashMap<String, Coord>> {
    let f = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut out: HashMap<String, Coord> = HashMap::new();
    for line in f.lines() {
        let s = line?;
        if s.trim().is_empty() || s.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = s.split_whitespace().collect();
        if t.len() < 4 {
            continue;
        }
        if out.contains_key(t[3]) {
            continue;
        }
        let start_1based = t[1].parse::<usize>().unwrap_or(0);
        out.insert(t[3].to_string(), Coord { start_1based });
    }
    Ok(out)
}

fn index_regions_offsets(path: &PathBuf) -> Result<HashMap<String, u64>> {
    let mut f = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut idx: HashMap<String, u64> = HashMap::new();
    let mut cur: Option<String> = None;
    let mut abs: u64 = 0;
    loop {
        let mut line = String::new();
        let n = f.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let this_start = abs;
        abs += n as u64;
        if line.starts_with('>') {
            cur = Some(region_key_from_header(line.trim_end()));
        } else if cur.is_some() && !line.trim().is_empty() {
            let key = cur.take().unwrap();
            idx.entry(key).or_insert(this_start);
        }
    }
    Ok(idx)
}

fn read_region_seq_at(path: &PathBuf, offset: u64) -> Result<Vec<u8>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    f.seek(SeekFrom::Start(offset))?;
    let mut r = BufReader::new(f);
    let mut seq = Vec::new();
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if line.starts_with('>') {
            break;
        }
        for &b in line.trim().as_bytes() {
            if !b.is_ascii_whitespace() {
                seq.push(b.to_ascii_uppercase());
            }
        }
    }
    Ok(seq)
}

fn process_one_molecule(
    seqs: &[(Vec<u8>, Vec<u8>)],
    region: &[u8],
    max_k: usize,
    stats: &mut SampleStats,
) {
    let mut per_pos_votes: Vec<Vec<u8>> = vec![Vec::new(); region.len()];

    for (seq, qual) in seqs {
        if seq.is_empty() || qual.is_empty() || seq.len() != qual.len() {
            continue;
        }
        if rand_aln_p(region.len(), seq.len()) > 0.0001 {
            continue;
        }
        let mut aligner = Aligner::with_capacity(seq.len(), region.len(), -5, -2, MatchParams::new(1, -3));
        let aln = aligner.local(seq, region);
        if aln
            .operations
            .iter()
            .any(|o| matches!(o, AlignmentOperation::Ins | AlignmentOperation::Del))
        {
            continue;
        }
        let mut qpos = aln.xstart;
        let mut rpos = aln.ystart;
        for op in &aln.operations {
            match op {
                AlignmentOperation::Match | AlignmentOperation::Subst => {
                    if qpos < seq.len() && rpos < region.len() {
                        let qb = seq[qpos].to_ascii_uppercase();
                        if phred(qual[qpos]) > 12 && matches!(qb, b'A' | b'C' | b'G' | b'T') {
                            per_pos_votes[rpos].push(qb);
                        }
                    }
                    qpos += 1;
                    rpos += 1;
                }
                AlignmentOperation::Xclip(l) => qpos += l,
                AlignmentOperation::Yclip(l) => rpos += l,
                AlignmentOperation::Ins => qpos += 1,
                AlignmentOperation::Del => rpos += 1,
            }
        }
    }

    for (i, votes) in per_pos_votes.iter().enumerate() {
        let ref_b = region[i];
        let Some(ref_i) = base_idx(ref_b) else {
            continue;
        };
        for k in 1..=max_k {
            if votes.len() < k {
                continue;
            }
            let cons = consensus_base(&votes[..k]);
            if !matches!(cons, b'A' | b'C' | b'G' | b'T') {
                continue;
            }
            let Some(cons_i) = base_idx(cons) else {
                continue;
            };
            stats.callable[k] += 1;
            stats.callable_by_ref[k][ref_i] += 1;
            if cons != ref_b {
                stats.mismatch[k] += 1;
                if let Some(si) = sub_idx(ref_i, cons_i) {
                    stats.mismatch_by_sub[k][si] += 1;
                }
            }
        }
    }
}

fn process_sample(stem: &str, max_k: usize) -> Result<SampleStats> {
    let bed = PathBuf::from(format!("{stem}_regions.bed"));
    let regions = PathBuf::from(format!("{stem}_reference.fa"));
    let split = PathBuf::from(format!("{stem}_split.fa"));
    for p in [&bed, &regions, &split] {
        if !p.is_file() {
            bail!("missing required file: {}", p.display());
        }
    }

    let bed_map = load_bed_first_coord(&bed)?;
    let reg_idx = index_regions_offsets(&regions)?;
    let mut stats = SampleStats::new(max_k);

    let f = BufReader::new(File::open(&split).with_context(|| format!("open {}", split.display()))?);
    let mut current_key: Option<String> = None;
    let mut current_segments: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut pending_seq: Option<Vec<u8>> = None;

    let flush = |key: &str,
                 segs: &[(Vec<u8>, Vec<u8>)],
                 stats: &mut SampleStats,
                 bed_map: &HashMap<String, Coord>,
                 reg_idx: &HashMap<String, u64>| -> Result<()> {
        if segs.is_empty() {
            return Ok(());
        }
        let Some(coord) = bed_map.get(key) else {
            return Ok(());
        };
        if coord.start_1based == 0 {
            return Ok(());
        }
        let Some(&off) = reg_idx.get(key) else {
            return Ok(());
        };
        let region = read_region_seq_at(&regions, off)?;
        if region.is_empty() {
            return Ok(());
        }
        process_one_molecule(segs, &region, max_k, stats);
        Ok(())
    };

    for line in f.lines() {
        let s = line?;
        let t = s.trim_end();
        if t.is_empty() || t.starts_with('+') {
            continue;
        }
        if t.starts_with('@') {
            let key = split_key_from_header(t);
            if current_key.as_ref() != Some(&key) {
                if let Some(old) = current_key.as_ref() {
                    flush(old, &current_segments, &mut stats, &bed_map, &reg_idx)?;
                }
                current_key = Some(key);
                current_segments.clear();
                pending_seq = None;
            }
            continue;
        }
        if pending_seq.is_none() {
            pending_seq = Some(t.as_bytes().to_vec());
        } else {
            let seq = pending_seq.take().unwrap_or_default();
            current_segments.push((seq, t.as_bytes().to_vec()));
        }
    }
    if let Some(old) = current_key.as_ref() {
        flush(old, &current_segments, &mut stats, &bed_map, &reg_idx)?;
    }
    Ok(stats)
}

fn resolve_output_path(p: &Path) -> PathBuf {
    if p.is_dir() {
        return p.join("incremental_repeat_error.tsv");
    }
    p.to_path_buf()
}

fn load_completed_samples(output_tsv: &Path, max_k: usize) -> Result<HashMap<String, usize>> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    if !output_tsv.is_file() {
        return Ok(seen);
    }
    let f = BufReader::new(File::open(output_tsv).with_context(|| format!("open {}", output_tsv.display()))?);
    for line in f.lines() {
        let s = line?;
        if s.trim().is_empty() || s.starts_with("sample\tk\t") {
            continue;
        }
        let t: Vec<&str> = s.split('\t').collect();
        if t.len() < 5 {
            continue;
        }
        let sample = t[0].to_string();
        if sample == "POOLED" {
            continue;
        }
        let Ok(k) = t[1].parse::<usize>() else {
            continue;
        };
        if k >= 1 && k <= max_k {
            let e = seen.entry(sample).or_insert(0);
            if k > *e {
                *e = k;
            }
        }
    }
    Ok(seen)
}

fn write_sample_rows(out: &mut File, sample: &str, s: &SampleStats, max_k: usize) -> Result<()> {
    for k in 1..=max_k {
        let c = s.callable[k];
        let m = s.mismatch[k];
        let r = if c > 0 { m as f64 / c as f64 } else { 0.0 };
        writeln!(out, "{sample}\t{k}\t{c}\t{m}\t{r:.10}")?;
    }
    out.flush()?;
    Ok(())
}

fn write_transition_rows(out: &mut File, sample: &str, s: &SampleStats, max_k: usize) -> Result<()> {
    for k in 1..=max_k {
        for (si, sub) in SUBSTITUTIONS.iter().enumerate() {
            let ref_i = match si {
                0..=2 => 0,
                3..=5 => 1,
                6..=8 => 2,
                _ => 3,
            };
            let callable_ref_n = s.callable_by_ref[k][ref_i];
            let mismatch_sub_n = s.mismatch_by_sub[k][si];
            let error_rate = if callable_ref_n > 0 {
                mismatch_sub_n as f64 / callable_ref_n as f64
            } else {
                0.0
            };
            writeln!(
                out,
                "{sample}\t{k}\t{sub}\t{callable_ref_n}\t{mismatch_sub_n}\t{error_rate:.10}"
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.max_k < 1 {
        bail!("--max-k must be >= 1");
    }

    let mut stems: Vec<String> = Vec::new();
    for entry in glob::glob(&args.split_glob).context("invalid --split-glob")? {
        let p = entry?;
        let s = p.to_string_lossy().to_string();
        let stem = s.strip_suffix("_split.fa").unwrap_or(&s).to_string();
        let b = std::path::Path::new(&stem)
            .file_name()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or_else(|| stem.clone());
        if !args.include_sample_substring.is_empty()
            && !b.to_lowercase().contains(&args.include_sample_substring.to_lowercase())
        {
            continue;
        }
        if !args.exclude_sample_substring.is_empty()
            && b.to_lowercase().contains(&args.exclude_sample_substring.to_lowercase())
        {
            continue;
        }
        stems.push(stem);
    }
    stems.sort();
    if stems.is_empty() {
        bail!("no samples matched --split-glob and filters");
    }

    let output_tsv = resolve_output_path(&args.output_tsv);
    if let Some(parent) = output_tsv.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output parent {}", parent.display()))?;
    }

    let done_counts = if args.resume {
        load_completed_samples(&output_tsv, args.max_k)?
    } else {
        HashMap::new()
    };
    let mut pending_stems: Vec<String> = Vec::new();
    for stem in &stems {
        let sample = Path::new(stem)
            .file_name()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or_else(|| stem.clone());
        let completed = done_counts.get(&sample).copied().unwrap_or(0);
        if args.resume && completed >= args.max_k {
            eprintln!("Skipping completed sample {sample}");
            continue;
        }
        pending_stems.push(stem.clone());
    }
    if pending_stems.is_empty() {
        bail!("no pending samples to process (all matched samples already complete?)");
    }

    let mut out = if args.resume && output_tsv.is_file() {
        File::options()
            .append(true)
            .open(&output_tsv)
            .with_context(|| format!("open {}", output_tsv.display()))?
    } else {
        let mut f = File::create(&output_tsv)
            .with_context(|| format!("create {}", output_tsv.display()))?;
        writeln!(f, "sample\tk\tcallable_n\tmismatch_n\terror_rate")?;
        f.flush()?;
        f
    };

    let transition_output_tsv = if args.transition_output_tsv.trim().is_empty() {
        None
    } else {
        Some(resolve_output_path(Path::new(&args.transition_output_tsv)))
    };
    let mut trans_out = if let Some(p) = transition_output_tsv.as_ref() {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create transition output parent {}", parent.display()))?;
        }
        if args.resume && p.is_file() {
            Some(
                File::options()
                    .append(true)
                    .open(p)
                    .with_context(|| format!("open {}", p.display()))?,
            )
        } else {
            let mut f = File::create(p).with_context(|| format!("create {}", p.display()))?;
            writeln!(
                f,
                "sample\tk\tsubstitution\tcallable_ref_n\tmismatch_sub_n\terror_rate"
            )?;
            f.flush()?;
            Some(f)
        }
    } else {
        None
    };

    let jobs = if args.jobs == 0 {
        thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        args.jobs
    }
    .max(1);
    eprintln!("Using {jobs} worker threads");

    let stems_arc = Arc::new(pending_stems);
    let next_idx = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::sync_channel::<(String, Result<SampleStats>)>(jobs * 2);

    thread::scope(|scope| -> Result<()> {
        for _ in 0..jobs {
            let tx = tx.clone();
            let stems_arc = Arc::clone(&stems_arc);
            let next_idx = Arc::clone(&next_idx);
            let max_k = args.max_k;
            scope.spawn(move || {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= stems_arc.len() {
                        break;
                    }
                    let stem = &stems_arc[idx];
                    let sample = Path::new(stem)
                        .file_name()
                        .map(|x| x.to_string_lossy().to_string())
                        .unwrap_or_else(|| stem.clone());
                    eprintln!("Processing {sample} ...");
                    let res = process_sample(stem, max_k);
                    if tx.send((sample, res)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut pooled = SampleStats::new(args.max_k);
        for (sample, res) in rx {
            let s = res?;
            write_sample_rows(&mut out, &sample, &s, args.max_k)?;
            if let Some(tf) = trans_out.as_mut() {
                write_transition_rows(tf, &sample, &s, args.max_k)?;
            }
            for k in 1..=args.max_k {
                pooled.callable[k] += s.callable[k];
                pooled.mismatch[k] += s.mismatch[k];
                for i in 0..4 {
                    pooled.callable_by_ref[k][i] += s.callable_by_ref[k][i];
                }
                for i in 0..12 {
                    pooled.mismatch_by_sub[k][i] += s.mismatch_by_sub[k][i];
                }
            }
            eprintln!("Flushed {sample}");
        }

        for k in 1..=args.max_k {
            let c = pooled.callable[k];
            let m = pooled.mismatch[k];
            let r = if c > 0 { m as f64 / c as f64 } else { 0.0 };
            writeln!(out, "POOLED\t{k}\t{c}\t{m}\t{r:.10}")?;
        }
        out.flush()?;
        if let Some(tf) = trans_out.as_mut() {
            write_transition_rows(tf, "POOLED", &pooled, args.max_k)?;
            tf.flush()?;
        }
        Ok(())
    })?;
    eprintln!("Wrote {}", output_tsv.display());
    Ok(())
}
