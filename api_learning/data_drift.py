import pandas as pd
from sklearn import datasets



reference_data = datasets.load_iris(as_frame='auto').frame

current_data = pd.read_csv('data/database.csv')

# standardize column names and dropping the timestamp column
current_data = current_data.drop('time', axis=1)
current_data.columns = reference_data.columns


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('reports/drift.html')

###


from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues
data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data)
data_test.as_dict()#['TestNumberOfMissingValues']['test_result']
