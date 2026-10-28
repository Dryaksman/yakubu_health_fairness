# Target and group definition (UCI Diabetes 130-US Hospitals)

- Target y = 1 if `readmitted` is `<30` (readmitted within 30 days), y = 0 if `NO` or `>30`.
- Prediction point: discharge. The original `readmitted` column is removed from the inputs.
- Protected attribute z = `race` (not an input). Main comparison: AfricanAmerican vs Caucasian.
- Excluded: encounters with discharge codes [11, 13, 14, 19, 20, 21] (expired / hospice).

Class counts in the cleaned cohort: y=1: 11314, y=0: 88029.
