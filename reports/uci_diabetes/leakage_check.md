# Leakage and timing check (discharge-time model)

| column | role | reason |
|---|---|---|
| encounter_id | excluded | identifier; never a predictor |
| patient_nbr | excluded (split key) | identifier; used only to keep a patient's encounters together |
| race | protected attribute z | kept outside the inputs; proxies among the inputs are checked later |
| gender | input | recorded during the stay or before; available at discharge |
| age | input (as midpoint) | known at admission |
| weight | excluded | about 97% missing |
| admission_type_id | input | recorded during the stay or before; available at discharge |
| discharge_disposition_id | input | known at discharge; death and hospice codes excluded from the cohort |
| admission_source_id | input | recorded during the stay or before; available at discharge |
| time_in_hospital | input | recorded during the stay or before; available at discharge |
| payer_code | input | recorded during the stay or before; available at discharge |
| medical_specialty | input | recorded during the stay or before; available at discharge |
| num_lab_procedures | input | recorded during the stay or before; available at discharge |
| num_procedures | input | recorded during the stay or before; available at discharge |
| num_medications | input | recorded during the stay or before; available at discharge |
| number_outpatient | input | recorded during the stay or before; available at discharge |
| number_emergency | input | recorded during the stay or before; available at discharge |
| number_inpatient | input | recorded during the stay or before; available at discharge |
| diag_1 | input (grouped) | assumed available at discharge; grouped into diagnosis categories |
| diag_2 | input (grouped) | assumed available at discharge; grouped into diagnosis categories |
| diag_3 | input (grouped) | assumed available at discharge; grouped into diagnosis categories |
| number_diagnoses | input | recorded during the stay or before; available at discharge |
| max_glu_serum | input | recorded during the stay or before; available at discharge |
| A1Cresult | input | recorded during the stay or before; available at discharge |
| metformin | input | recorded during the stay or before; available at discharge |
| repaglinide | input | recorded during the stay or before; available at discharge |
| nateglinide | input | recorded during the stay or before; available at discharge |
| chlorpropamide | input | recorded during the stay or before; available at discharge |
| glimepiride | input | recorded during the stay or before; available at discharge |
| acetohexamide | input | recorded during the stay or before; available at discharge |
| glipizide | input | recorded during the stay or before; available at discharge |
| glyburide | input | recorded during the stay or before; available at discharge |
| tolbutamide | input | recorded during the stay or before; available at discharge |
| pioglitazone | input | recorded during the stay or before; available at discharge |
| rosiglitazone | input | recorded during the stay or before; available at discharge |
| acarbose | input | recorded during the stay or before; available at discharge |
| miglitol | input | recorded during the stay or before; available at discharge |
| troglitazone | input | recorded during the stay or before; available at discharge |
| tolazamide | input | recorded during the stay or before; available at discharge |
| examide | input | recorded during the stay or before; available at discharge |
| citoglipton | input | recorded during the stay or before; available at discharge |
| insulin | input | recorded during the stay or before; available at discharge |
| glyburide-metformin | input | recorded during the stay or before; available at discharge |
| glipizide-metformin | input | recorded during the stay or before; available at discharge |
| glimepiride-pioglitazone | input | recorded during the stay or before; available at discharge |
| metformin-rosiglitazone | input | recorded during the stay or before; available at discharge |
| metformin-pioglitazone | input | recorded during the stay or before; available at discharge |
| change | input | recorded during the stay or before; available at discharge |
| diabetesMed | input | recorded during the stay or before; available at discharge |
| readmitted | target source | defines y; removed from the inputs |
