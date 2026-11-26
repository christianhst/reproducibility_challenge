# How to run our code
To be able to run our code and see how we validated their claims and reproduced their results, run the following lines.
#1 create the virtual enviroment.
`python3.10 -m venv .venv`
`source .venv/bin/activate`
`python -m pip install -r requirements.txt`

#2 Run the preprocessing and models.
Create the BCICIV_2a_gdf directory to store the downloaded BCI IV-2a dataset.
Create the mymat_raw directory to store the preprocessings

Run the model of your choice
mkdir -p logs_<model_name>/validation_<version>
for r in 1 2 3 4 5; do
  caffeinate -i python3 -u <model_name>.py 2>&1 | tee logs_<model_name>/validation_<version>/run$r.txt
done

Note:  val_data_list.append(img[-number_validate:]) and val_label_list.append(label[-number_validate:]) is validation 2 while val_data_list.append(img[number_validate:]) and val_label_list.append(label[number_validate:]) is validation 1 (in the main_subject_specific.py (line 464 and 465) change depending on what type of validation you want to do)

#3 Summarize results
run reproduced_model_trainings/summarize_runs.py (change folder directory for each logs_<model_name>)
create repro_results.xlsx to manually note your findings
like this
|  | Paper,val1 | Repro,val1 | Repro,val2 | std,val1 | SE,val1 | std,val2| SE,val2|
|DeepConvNet|-----|--------|------|-----|--------|------|-----|
|ShallowConvNet|-----|--------|------|-----|--------|------|-----|
|CTNet|-----|--------|------|-----|--------|------|-----|
|Conformer|-----|--------|------|-----|--------|------|-----|
|EEGNet|-----|--------|------|-----|--------|------|-----|

then you can run `python3 repro_barplot.py`
