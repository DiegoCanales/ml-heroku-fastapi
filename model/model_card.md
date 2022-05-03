# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Diego Canales created the model. It is support-vector machines using the default hyperparameters in scikit-learn 1.0.2.

## Intended Use

This model should be used to classify the segment salary of person based on some census attributes.

## Training Data

The data was obtained from https://archive.ics.uci.edu/ml/datasets/census+income. The training is done using 80% of this data.

## Evaluation Data

The data was obtained from https://archive.ics.uci.edu/ml/datasets/census+income. The training is done using 20% of this data.

## Metrics

The metrics used are precision, recall, fbeta. The values of general metrics of the model are presented in the following table.

| precision | recall | fbeta |
|-----------|--------|-------|
| 0.756     | 0.237  | 0.360 |

The slice performance can be found in `metrics/slice_performance.csv`.

## Ethical Considerations

The data contains sensitive information about the person like rage, gender and origin country. Because of that the model can be discriminative.

## Caveats and Recommendations

To avoid discrimination based on sensitive data, experiments can be carried out avoiding this features.
