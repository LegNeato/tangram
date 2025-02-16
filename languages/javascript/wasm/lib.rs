// wasm_bindgen generates some code in their proc macro that contains unused unit expressions.
// This has been fixed in https://github.com/rustwasm/wasm-bindgen/issues/2774 , but not yet
// released.
#![allow(clippy::unused_unit)]

use std::collections::BTreeMap;
use wasm_bindgen::{prelude::*, JsValue};

#[wasm_bindgen(js_name = "loadModelFromArrayBuffer")]
pub fn load_model_from_array_buffer(bytes: JsValue) -> Result<Model, JsValue> {
	let bytes: serde_bytes::ByteBuf =
		serde_wasm_bindgen::from_value(bytes).map_err(|e| e.to_string())?;
	let model = tangram_model::from_bytes(&bytes).map_err(|e| e.to_string())?;
	let model = tangram_core::predict::Model::from(model);
	let model = Model(model);
	Ok(model)
}

#[wasm_bindgen(js_name = "modelId")]
pub fn model_id(model: &Model) -> Result<String, JsValue> {
	Ok(model.0.id.to_string())
}

#[wasm_bindgen]
pub fn predict(model: &Model, input: JsValue, options: JsValue) -> Result<JsValue, JsValue> {
	let input: PredictInputSingleOrMultiple = input.into_serde().map_err(|e| e.to_string())?;
	let options: Option<PredictOptions> = options.into_serde().map_err(|e| e.to_string())?;
	let options = options.map(Into::into).unwrap_or_default();
	let model = &model.0;
	match input {
		PredictInputSingleOrMultiple::Single(input) => {
			let input = input.into();
			let mut output = tangram_core::predict::predict(model, &[input], &options);
			let output = output.remove(0);
			let output = output.into();
			let output = PredictOutputSingleOrMultiple::Single(output);
			let output = JsValue::from_serde(&output).map_err(|e| e.to_string())?;
			Ok(output)
		}
		PredictInputSingleOrMultiple::Multiple(input) => {
			let input = input.into_iter().map(Into::into).collect::<Vec<_>>();
			let output = tangram_core::predict::predict(model, &input, &options);
			let output = output.into_iter().map(Into::into).collect();
			let output = PredictOutputSingleOrMultiple::Multiple(output);
			let output = JsValue::from_serde(&output).map_err(|e| e.to_string())?;
			Ok(output)
		}
	}
}

#[wasm_bindgen]
pub struct Model(tangram_core::predict::Model);

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum PredictInputSingleOrMultiple {
	Single(PredictInput),
	Multiple(PredictInputMultiple),
}

#[derive(serde::Deserialize)]
struct PredictInput(pub BTreeMap<String, PredictInputValue>);

type PredictInputMultiple = Vec<PredictInput>;

impl From<PredictInput> for tangram_core::predict::PredictInput {
	fn from(value: PredictInput) -> tangram_core::predict::PredictInput {
		tangram_core::predict::PredictInput(
			value
				.0
				.into_iter()
				.map(|(key, value)| (key, value.into()))
				.collect(),
		)
	}
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum PredictInputValue {
	Number(f64),
	String(String),
}

impl From<PredictInputValue> for tangram_core::predict::PredictInputValue {
	fn from(value: PredictInputValue) -> tangram_core::predict::PredictInputValue {
		match value {
			PredictInputValue::Number(value) => {
				tangram_core::predict::PredictInputValue::Number(value)
			}
			PredictInputValue::String(value) => {
				tangram_core::predict::PredictInputValue::String(value)
			}
		}
	}
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PredictOptions {
	pub threshold: Option<f32>,
	pub compute_feature_contributions: Option<bool>,
}

impl From<PredictOptions> for tangram_core::predict::PredictOptions {
	fn from(value: PredictOptions) -> tangram_core::predict::PredictOptions {
		let mut options = tangram_core::predict::PredictOptions::default();
		if let Some(threshold) = value.threshold {
			options.threshold = threshold;
		}
		if let Some(compute_feature_contributions) = value.compute_feature_contributions {
			options.compute_feature_contributions = compute_feature_contributions;
		}
		options
	}
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum PredictOutputSingleOrMultiple {
	Single(PredictOutput),
	Multiple(PredictOutputMultiple),
}

#[derive(serde::Serialize)]
#[serde(tag = "type")]
enum PredictOutput {
	#[serde(rename = "regression")]
	Regression(RegressionPredictOutput),
	#[serde(rename = "binary_classification")]
	BinaryClassification(BinaryClassificationPredictOutput),
	#[serde(rename = "multiclass_classification")]
	MulticlassClassification(MulticlassClassificationPredictOutput),
}

type PredictOutputMultiple = Vec<PredictOutput>;

impl From<tangram_core::predict::PredictOutput> for PredictOutput {
	fn from(value: tangram_core::predict::PredictOutput) -> Self {
		match value {
			tangram_core::predict::PredictOutput::Regression(value) => {
				PredictOutput::Regression(value.into())
			}
			tangram_core::predict::PredictOutput::BinaryClassification(value) => {
				PredictOutput::BinaryClassification(value.into())
			}
			tangram_core::predict::PredictOutput::MulticlassClassification(value) => {
				PredictOutput::MulticlassClassification(value.into())
			}
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct RegressionPredictOutput {
	pub value: f32,
	pub feature_contributions: Option<FeatureContributions>,
}

impl From<tangram_core::predict::RegressionPredictOutput> for RegressionPredictOutput {
	fn from(value: tangram_core::predict::RegressionPredictOutput) -> Self {
		RegressionPredictOutput {
			value: value.value,
			feature_contributions: value.feature_contributions.map(Into::into),
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct BinaryClassificationPredictOutput {
	pub class_name: String,
	pub probability: f32,
	pub feature_contributions: Option<FeatureContributions>,
}

impl From<tangram_core::predict::BinaryClassificationPredictOutput>
	for BinaryClassificationPredictOutput
{
	fn from(value: tangram_core::predict::BinaryClassificationPredictOutput) -> Self {
		BinaryClassificationPredictOutput {
			class_name: value.class_name,
			probability: value.probability,
			feature_contributions: value.feature_contributions.map(Into::into),
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct MulticlassClassificationPredictOutput {
	pub class_name: String,
	pub probability: f32,
	pub probabilities: BTreeMap<String, f32>,
	pub feature_contributions: Option<BTreeMap<String, FeatureContributions>>,
}

impl From<tangram_core::predict::MulticlassClassificationPredictOutput>
	for MulticlassClassificationPredictOutput
{
	fn from(value: tangram_core::predict::MulticlassClassificationPredictOutput) -> Self {
		MulticlassClassificationPredictOutput {
			class_name: value.class_name,
			probability: value.probability,
			probabilities: value.probabilities,
			feature_contributions: value.feature_contributions.map(|feature_contributions| {
				feature_contributions
					.into_iter()
					.map(|(key, value)| (key, value.into()))
					.collect()
			}),
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeatureContributions {
	pub baseline_value: f32,
	pub output_value: f32,
	pub entries: Vec<FeatureContributionEntry>,
}

impl From<tangram_core::predict::FeatureContributions> for FeatureContributions {
	fn from(value: tangram_core::predict::FeatureContributions) -> Self {
		FeatureContributions {
			baseline_value: value.baseline_value,
			output_value: value.output_value,
			entries: value.entries.into_iter().map(Into::into).collect(),
		}
	}
}

#[derive(serde::Serialize)]
#[serde(tag = "type")]
enum FeatureContributionEntry {
	#[serde(rename = "identity")]
	Identity(IdentityFeatureContribution),
	#[serde(rename = "normalized")]
	Normalized(NormalizedFeatureContribution),
	#[serde(rename = "one_hot_encoded")]
	OneHotEncoded(OneHotEncodedFeatureContribution),
	#[serde(rename = "bag_of_words")]
	BagOfWords(BagOfWordsFeatureContribution),
	#[serde(rename = "bag_of_words_cosine_similarity")]
	BagOfWordsCosineSimilarity(BagOfWordsCosineSimilarityFeatureContribution),
	#[serde(rename = "word_embedding")]
	WordEmbedding(WordEmbeddingFeatureContribution),
}

impl From<tangram_core::predict::FeatureContributionEntry> for FeatureContributionEntry {
	fn from(value: tangram_core::predict::FeatureContributionEntry) -> Self {
		match value {
			tangram_core::predict::FeatureContributionEntry::Identity(value) => {
				FeatureContributionEntry::Identity(value.into())
			}
			tangram_core::predict::FeatureContributionEntry::Normalized(value) => {
				FeatureContributionEntry::Normalized(value.into())
			}
			tangram_core::predict::FeatureContributionEntry::OneHotEncoded(value) => {
				FeatureContributionEntry::OneHotEncoded(value.into())
			}
			tangram_core::predict::FeatureContributionEntry::BagOfWords(value) => {
				FeatureContributionEntry::BagOfWords(value.into())
			}
			tangram_core::predict::FeatureContributionEntry::BagOfWordsCosineSimilarity(value) => {
				FeatureContributionEntry::BagOfWordsCosineSimilarity(value.into())
			}
			tangram_core::predict::FeatureContributionEntry::WordEmbedding(value) => {
				FeatureContributionEntry::WordEmbedding(value.into())
			}
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct IdentityFeatureContribution {
	column_name: String,
	feature_contribution_value: f32,
	feature_value: f32,
}

impl From<tangram_core::predict::IdentityFeatureContribution> for IdentityFeatureContribution {
	fn from(value: tangram_core::predict::IdentityFeatureContribution) -> Self {
		IdentityFeatureContribution {
			column_name: value.column_name,
			feature_contribution_value: value.feature_contribution_value,
			feature_value: value.feature_value,
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct NormalizedFeatureContribution {
	column_name: String,
	feature_contribution_value: f32,
}

impl From<tangram_core::predict::NormalizedFeatureContribution> for NormalizedFeatureContribution {
	fn from(value: tangram_core::predict::NormalizedFeatureContribution) -> Self {
		NormalizedFeatureContribution {
			column_name: value.column_name,
			feature_contribution_value: value.feature_contribution_value,
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct OneHotEncodedFeatureContribution {
	column_name: String,
	variant: Option<String>,
	feature_value: bool,
	feature_contribution_value: f32,
}

impl From<tangram_core::predict::OneHotEncodedFeatureContribution>
	for OneHotEncodedFeatureContribution
{
	fn from(value: tangram_core::predict::OneHotEncodedFeatureContribution) -> Self {
		OneHotEncodedFeatureContribution {
			column_name: value.column_name,
			variant: value.variant,
			feature_value: value.feature_value,
			feature_contribution_value: value.feature_contribution_value,
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct BagOfWordsFeatureContribution {
	column_name: String,
	ngram: NGram,
	feature_value: f32,
	feature_contribution_value: f32,
}

impl From<tangram_core::predict::BagOfWordsFeatureContribution> for BagOfWordsFeatureContribution {
	fn from(value: tangram_core::predict::BagOfWordsFeatureContribution) -> Self {
		BagOfWordsFeatureContribution {
			column_name: value.column_name,
			ngram: value.ngram.into(),
			feature_value: value.feature_value,
			feature_contribution_value: value.feature_contribution_value,
		}
	}
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum NGram {
	Unigram(String),
	Bigram(String, String),
}

impl From<tangram_core::predict::NGram> for NGram {
	fn from(value: tangram_core::predict::NGram) -> Self {
		match value {
			tangram_core::predict::NGram::Unigram(token) => NGram::Unigram(token),
			tangram_core::predict::NGram::Bigram(token_a, token_b) => {
				NGram::Bigram(token_a, token_b)
			}
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct BagOfWordsCosineSimilarityFeatureContribution {
	column_name_a: String,
	column_name_b: String,
	feature_value: f32,
	feature_contribution_value: f32,
}

impl From<tangram_core::predict::BagOfWordsCosineSimilarityFeatureContribution>
	for BagOfWordsCosineSimilarityFeatureContribution
{
	fn from(value: tangram_core::predict::BagOfWordsCosineSimilarityFeatureContribution) -> Self {
		BagOfWordsCosineSimilarityFeatureContribution {
			column_name_a: value.column_name_a,
			column_name_b: value.column_name_b,
			feature_value: value.feature_value,
			feature_contribution_value: value.feature_contribution_value,
		}
	}
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WordEmbeddingFeatureContribution {
	column_name: String,
	value_index: usize,
	feature_contribution_value: f32,
}

impl From<tangram_core::predict::WordEmbeddingFeatureContribution>
	for WordEmbeddingFeatureContribution
{
	fn from(value: tangram_core::predict::WordEmbeddingFeatureContribution) -> Self {
		WordEmbeddingFeatureContribution {
			column_name: value.column_name,
			value_index: value.value_index,
			feature_contribution_value: value.feature_contribution_value,
		}
	}
}
