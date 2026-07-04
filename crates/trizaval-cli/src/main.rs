//! trizaval CLI — runs trizaval-core statistical routines from the
//! command line, designed to drop into CI pipelines.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use trizaval_core::bootstrap::{self, BootstrapResult};
use trizaval_core::correction::{self, CorrectionMethod, CorrectionResult};
use trizaval_core::effect_size::{self, EffectSizeResult};
use trizaval_core::judge_calibration::{self, LengthBiasResult, PairwiseDebiasResult, Preference};
use trizaval_core::sequential::{SequentialDecision, SequentialTest};

#[derive(Parser)]
#[command(
    name = "trizaval",
    version,
    about = "Statistically rigorous evaluation tooling for AI systems"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute a block-bootstrap confidence interval for the mean of
    /// a set of evaluation scores.
    Bootstrap {
        /// Path to a JSON file containing an array of numbers.
        #[arg(long)]
        input: PathBuf,

        #[arg(long, default_value_t = 1)]
        block_size: usize,

        #[arg(long, default_value_t = 2000)]
        n_resamples: usize,

        #[arg(long, default_value_t = 0.95)]
        confidence_level: f64,

        #[arg(long)]
        seed: Option<u64>,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Run a sequential hypothesis test over a series of
    /// observations, stopping as soon as significance is reached.
    Sequential {
        /// Path to a JSON file containing an array of numbers, fed
        /// to the test in order.
        #[arg(long)]
        input: PathBuf,

        /// Desired Type I error rate.
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,

        /// Prior standard deviation on the effect size worth
        /// detecting.
        #[arg(long, default_value_t = 0.5)]
        tau: f64,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Apply a multiple-comparisons correction to a set of p-values.
    Correction {
        /// Path to a JSON file containing an array of p-values.
        #[arg(long)]
        input: PathBuf,

        #[arg(long, default_value_t = 0.05)]
        alpha: f64,

        #[arg(long, value_enum, default_value_t = CliCorrectionMethod::BenjaminiHochberg)]
        method: CliCorrectionMethod,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Compute Cohen's d / Hedges' g effect size between two groups
    /// of evaluation scores.
    EffectSize {
        /// Path to a JSON file containing an array of baseline scores.
        #[arg(long)]
        baseline: PathBuf,

        /// Path to a JSON file containing an array of treatment scores.
        #[arg(long)]
        treatment: PathBuf,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Fit and remove length bias from a set of LLM-judge scores.
    JudgeLengthBias {
        /// Path to a JSON file containing an array of judge scores.
        #[arg(long)]
        scores: PathBuf,

        /// Path to a JSON file containing an array of response
        /// lengths, same order and length as `scores`.
        #[arg(long)]
        lengths: PathBuf,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },

    /// Debias a single pairwise A/B judge preference using two
    /// judgments made with positions swapped.
    JudgePairwise {
        /// Preference recorded when response A was shown first.
        #[arg(long, value_enum)]
        original_order: CliPreference,

        /// Preference recorded when response B was shown first
        /// (expressed in terms of the same underlying A/B identity).
        #[arg(long, value_enum)]
        swapped_order: CliPreference,

        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, ValueEnum)]
enum CliCorrectionMethod {
    Bonferroni,
    BenjaminiHochberg,
}

impl From<CliCorrectionMethod> for CorrectionMethod {
    fn from(m: CliCorrectionMethod) -> Self {
        match m {
            CliCorrectionMethod::Bonferroni => CorrectionMethod::Bonferroni,
            CliCorrectionMethod::BenjaminiHochberg => CorrectionMethod::BenjaminiHochberg,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum CliPreference {
    PrefersA,
    PrefersB,
    Tie,
}

impl From<CliPreference> for Preference {
    fn from(p: CliPreference) -> Self {
        match p {
            CliPreference::PrefersA => Preference::PrefersA,
            CliPreference::PrefersB => Preference::PrefersB,
            CliPreference::Tie => Preference::Tie,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Bootstrap {
            input,
            block_size,
            n_resamples,
            confidence_level,
            seed,
            format,
        } => run_bootstrap(input, block_size, n_resamples, confidence_level, seed, format),

        Commands::Sequential {
            input,
            alpha,
            tau,
            format,
        } => run_sequential(input, alpha, tau, format),

        Commands::Correction {
            input,
            alpha,
            method,
            format,
        } => run_correction(input, alpha, method, format),

        Commands::EffectSize {
            baseline,
            treatment,
            format,
        } => run_effect_size(baseline, treatment, format),

        Commands::JudgeLengthBias {
            scores,
            lengths,
            format,
        } => run_judge_length_bias(scores, lengths, format),

        Commands::JudgePairwise {
            original_order,
            swapped_order,
            format,
        } => run_judge_pairwise(original_order, swapped_order, format),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn read_json_numbers(path: &PathBuf) -> Result<Vec<f64>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse {} as a JSON array of numbers: {e}", path.display()))
}

fn run_bootstrap(
    input: PathBuf,
    block_size: usize,
    n_resamples: usize,
    confidence_level: f64,
    seed: Option<u64>,
    format: OutputFormat,
) -> Result<(), String> {
    let data = read_json_numbers(&input)?;

    let result = bootstrap::block_bootstrap(
        &data,
        block_size,
        n_resamples,
        confidence_level,
        bootstrap::mean,
        seed,
    )
    .map_err(|e| e.to_string())?;

    match format {
        OutputFormat::Text => print_bootstrap_text(&result),
        OutputFormat::Json => print_json(&serde_json::json!({
            "point_estimate": result.point_estimate,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "confidence_level": result.confidence_level,
            "n_resamples": result.n_resamples,
        })),
    }

    Ok(())
}

fn run_sequential(input: PathBuf, alpha: f64, tau: f64, format: OutputFormat) -> Result<(), String> {
    let data = read_json_numbers(&input)?;

    let mut test = SequentialTest::new(alpha, tau).map_err(|e| e.to_string())?;

    let mut rejected_at: Option<usize> = None;
    for &x in &data {
        let update = test.update(x);
        if update.decision == SequentialDecision::RejectNull {
            rejected_at = Some(update.n);
            break;
        }
    }

    match format {
        OutputFormat::Text => {
            match rejected_at {
                Some(n) => println!("Null rejected at n = {n}"),
                None => println!("No rejection after {} observations", data.len()),
            }
            println!("Final mean:          {:.6}", test.current_mean());
            println!("Final n:             {}", test.n());
        }
        OutputFormat::Json => print_json(&serde_json::json!({
            "rejected_at_n": rejected_at,
            "final_mean": test.current_mean(),
            "final_n": test.n(),
            "total_observations": data.len(),
        })),
    }

    Ok(())
}

fn run_correction(
    input: PathBuf,
    alpha: f64,
    method: CliCorrectionMethod,
    format: OutputFormat,
) -> Result<(), String> {
    let p_values = read_json_numbers(&input)?;

    let result: CorrectionResult = correction::correct_p_values(&p_values, alpha, method.into())
        .map_err(|e| e.to_string())?;

    match format {
        OutputFormat::Text => {
            println!("Method:              {:?}", result.method);
            println!("Alpha:               {}", result.alpha);
            println!("Adjusted p-values:   {:?}", result.adjusted_p_values);
            println!("Rejected:            {:?}", result.rejected);
            println!("Total rejections:    {}", result.rejected.iter().filter(|&&r| r).count());
        }
        OutputFormat::Json => print_json(&serde_json::json!({
            "adjusted_p_values": result.adjusted_p_values,
            "rejected": result.rejected,
            "alpha": result.alpha,
        })),
    }

    Ok(())
}

fn run_effect_size(baseline: PathBuf, treatment: PathBuf, format: OutputFormat) -> Result<(), String> {
    let baseline_data = read_json_numbers(&baseline)?;
    let treatment_data = read_json_numbers(&treatment)?;

    let result: EffectSizeResult =
        effect_size::cohens_d(&baseline_data, &treatment_data).map_err(|e| e.to_string())?;

    match format {
        OutputFormat::Text => {
            println!("Cohen's d:           {:.6}", result.cohens_d);
            println!("Hedges' g:           {:.6}", result.hedges_g);
            println!("Magnitude:           {:?}", result.magnitude);
            println!("n_baseline:          {}", result.n_baseline);
            println!("n_treatment:         {}", result.n_treatment);
        }
        OutputFormat::Json => print_json(&serde_json::json!({
            "cohens_d": result.cohens_d,
            "hedges_g": result.hedges_g,
            "magnitude": format!("{:?}", result.magnitude),
            "n_baseline": result.n_baseline,
            "n_treatment": result.n_treatment,
        })),
    }

    Ok(())
}

fn run_judge_length_bias(scores: PathBuf, lengths: PathBuf, format: OutputFormat) -> Result<(), String> {
    let scores_data = read_json_numbers(&scores)?;
    let lengths_data = read_json_numbers(&lengths)?;

    let result: LengthBiasResult =
        judge_calibration::length_bias_correction(&scores_data, &lengths_data).map_err(|e| e.to_string())?;

    match format {
        OutputFormat::Text => {
            println!("Slope:               {:.6}", result.slope);
            println!("Intercept:           {:.6}", result.intercept);
            println!("Correlation:         {:.6}", result.correlation);
            println!("Adjusted scores:     {:?}", result.adjusted_scores);
        }
        OutputFormat::Json => print_json(&serde_json::json!({
            "slope": result.slope,
            "intercept": result.intercept,
            "correlation": result.correlation,
            "adjusted_scores": result.adjusted_scores,
        })),
    }

    Ok(())
}

fn run_judge_pairwise(
    original_order: CliPreference,
    swapped_order: CliPreference,
    format: OutputFormat,
) -> Result<(), String> {
    let result: PairwiseDebiasResult =
        judge_calibration::debias_pairwise_judgment(original_order.into(), swapped_order.into());

    match format {
        OutputFormat::Text => {
            println!("Preference:              {:?}", result.preference);
            println!("Position bias detected:  {}", result.position_bias_detected);
        }
        OutputFormat::Json => print_json(&serde_json::json!({
            "preference": format!("{:?}", result.preference),
            "position_bias_detected": result.position_bias_detected,
        })),
    }

    Ok(())
}

fn print_bootstrap_text(result: &BootstrapResult) {
    println!("Point estimate:     {:.6}", result.point_estimate);
    println!(
        "{:.0}% CI:            [{:.6}, {:.6}]",
        result.confidence_level * 100.0,
        result.ci_lower,
        result.ci_upper
    );
    println!("Resamples:          {}", result.n_resamples);
}

fn print_json(value: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
}