import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data_file = "student_performance_prediction.csv"
df = pd.read_csv(data_file)
df = df.drop(columns=['Student ID'])

df['Study Hours per Week']  = [-i if i<=0 else i for i in df['Study Hours per Week']]

df['Attendance Rate'] = [-i if i<=0 else i for i in df['Attendance Rate']]

df['Attendance Rate'] = [i - 50 if i > 100 else i for i in df['Attendance Rate']]

df['Previous Grades'] = [i - 100 if i > 100 else i for i in df['Previous Grades']]

df['Participation in Extracurricular Activities'] = df['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})

df['Parent Education Level'] = df['Parent Education Level'].map({'High School': 0, 'Bachelor': 1, 'Master': 2, 'Associate': 3, 'Doctorate': 4})

df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})

df = df.interpolate(method ='linear', limit_direction ='forward')

df['Attendance Rate'].fillna(df['Attendance Rate'].mean(), inplace=True)

df['Participation in Extracurricular Activities'] = df['Participation in Extracurricular Activities'].round()

df['Parent Education Level'] = df['Parent Education Level'].round()

df['Passed'] = df['Passed'].round()

scaler_0_1 = MinMaxScaler(feature_range=(0, 1))

df[['Study Hours per Week', 'Attendance Rate']] = scaler_0_1.fit_transform(df[['Study Hours per Week', 'Attendance Rate']])

scaler_0_4 = MinMaxScaler(feature_range=(0, 4))

df[['Previous Grades']] = scaler_0_4.fit_transform(df[['Previous Grades']])

print(df.head(10))  



