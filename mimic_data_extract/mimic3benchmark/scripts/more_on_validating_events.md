# More on validating results

The `validate_events.py` tries to assert some assumptions about the data; find events with specific problems and fix these problems if possible.  
Assumptions we assert:
* There is one-to-one mapping between HADM_ID and ICUSTAY_ID in `stays.csv` files.
* HADM_ID and ICUSTAY_ID are not empty in `stays.csv` files.
* `stays.csv` and `events.csv` files are always present.
* There is no case, where after initial filtering we cannot recover empty ICUSTAY_IDs.
  
Problems we fix (the order of this steps is fixed):
* Remove all events for which HADM_ID is missing.
* Remove all events for which HADM_ID is not present in `stays.csv`.
* If ICUSTAY_ID is missing in an event and HADM_ID is not missing, then we look at `stays.csv` and try to recover ICUSTAY_ID.
* Remove all events for which we cannot recover ICUSTAY_ID.
* Remove all events for which ICUSTAY_ID is not present in `stays.csv`.
