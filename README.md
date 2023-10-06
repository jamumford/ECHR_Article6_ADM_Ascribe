
## README for Annotation Datasets for ECtHR Article 6 Cases

### Overview

This repository contains datasets detailing annotations for cases related to Article 6 from the European Court of Human Rights (ECtHR). Annotations have been provided by individual annotators and are also summarised for easier analysis.

### Related Publications

If you use the datasets provided in this repository, or if you want to gain a deeper understanding of the context and methods behind the creation of these datasets, please refer to the following paper:

**Combining a Legal Knowledge Model with Machine Learning for Reasoning with Legal Cases**

Citation Format:
Mumford J, Atkinson K, Bench-Capon T. (2023). Combining a Legal Knowledge Model with Machine Learning for Reasoning with Legal Cases. In Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law. https://doi.org/10.1145/3594536.3595158


### Repository Structure

```
.
├── final_json_outputs/
│   ├── domain/
│   │   ├── [anonymised participant id]/
│   │   │   ├── [anonymised]_article6_non_cases.json
│   │   │   └── [anonymised]_article6_vio_cases.json
│   └── general/
│       ├── [anonymised participant id]/
│       │   ├── [anonymised]_article6_non_cases.json
│       │   └── [anonymised]_article6_vio_cases.json
└── Analysis/
    ├── Domain_non_stats.json
    ├── Domain_vio_stats.json
    ├── General_non_stats.json
    ├── General_vio_stats.json
    ├── Overall_non_stats.json
    └── Overall_vio_stats.json
```

### File Descriptions

1. **Individual Annotator Files**:
    - Location: `final_json_outputs/[domain or general]/[anonymised participant id]/`
    - Contains 54 sets of annotations, with 2 json files per annotator.
    - Each JSON file represents a case with associated annotations, provided as key-value pairs.
    - Example File: `3aub7_article6_non_cases.json`

2. **Summary Annotation Files**:
    - Location: `Analysis/`
    - Contains 6 JSON files that provide summarised annotations.
    - Each JSON file aggregates data from multiple cases and presents them as key-value pairs.
    - Example File: `Overall_non_stats.json`

### Data Format

- Each JSON object in the file represents a single case.
- Annotations are given as key-value pairs where:
    - Key: Aspect of the case (e.g., "CRIMINAL LIMB?", "FLAGGED?", "FAIR", etc.) taken from ADM (Angelic Domain Model) of Article 6 detailed in the associated paper (https://doi.org/10.1145/3594536.3595158). Two exception are the keys (1) CRIMINAL LIMB? and (2) FLAGGED?, which are not nodes in the ADM but report the (1) relevant limb of the case (criminal, civil, or both) and (2) if the annotator wishes to flag the case (note that the reasons for flagging are not provided in the datasets).
    - Value: Annotation counts expressed in a tuple [positive ascription, negative ascription, no ascription], where positive (respectively negative) ascription indicates that the key was deemed relevant to the case and its requirements were satisfied (respectively unsatisfied), and no ascription indicates the key was not deemed relevant to the case. Note an exception exists for key CRIMINAL LIMB?, which has value tuple [yes, no, both], where 'no' indicates the case relates to the civil limb, and 'both' indicates both limbs were deemed to apply. Each individual annotator's json file will have a tuple that sums to 1 (since all ascription classes for any given key are mutually exclusive), whereas each tuple in a summary json file will have a sum equal to the number of annotators that reviewed the legal case relevant to the JSON object in the file.
  
### Example

For a single case, the data might look like:

```json
{
    "001-187689": {
        "CRIMINAL LIMB?": [26, 0, 1],
        "FLAGGED?": [2, 0, 25],
        "FAIR:" [24, 3, 0],
        ...
    }
}
```

Which indicates that for legal case 001-187689, 27 annotators reviewed the case, 26 ascribed it as a criminal limb case and 1 ascribed it as both criminal and civil, 2 positively ascribed the flag marker and 25 did not flag the case, 24 positively ascribed and 3 negatively ascribed the FAIR issue to the case.

### Usage

To analyse the data, you can parse the JSON files using libraries like `json` in Python and extract the necessary information. Make sure to handle multiple JSON objects in the summary files.

### License

Please refer to the `LICENSE` file in the repository for usage rights and restrictions.

### Acknowledgements

We would like to thank all the annotators who contributed to this dataset, providing invaluable insights into the cases related to Article 6 from the ECtHR.
