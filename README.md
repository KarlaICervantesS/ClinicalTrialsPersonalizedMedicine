Genetic Algorithm App for Subject Allocation

App link
https://clinicaltrialspersonalizedmedicine.streamlit.app/

Purpose
This Streamlit app assigns subjects to groups using a genetic algorithm based on uploaded covariate data.

Input file requirements
The app accepts:
- CSV
- XLSX
- XLS

Your file must follow this structure:
1. First column: subject ID
2. Next columns: covariates used in the model

Important for categorical variables
- Categorical covariate columns must be binary
- They must already be coded as -1 and 1 only
- Do not use text labels such as Male/Female or Yes/No
- Do not use 0 and 1
- Do not use variables with more than two categories

Example

ID,age,bmi,sex,smoker
1,54,28.1,-1,1
2,61,31.4,1,-1
3,49,26.8,-1,-1

In this example:
- age and bmi are quantitative covariates
- sex and smoker are categorical covariates coded as -1 and 1

How to use the app
1. Open the app:
   https://clinicaltrialspersonalizedmedicine.streamlit.app/

2. Upload your data file:
   - CSV, XLSX, or XLS
   - If using Excel and your data is not in the first sheet, type the sheet name

3. Enter the study settings:
   - Number of subjects
   - Number of groups
   - Number of quantitative covariates
   - Number of categorical covariates

4. Check the data preview and input summary shown by the app

5. Click:
   Run algorithm

6. Review the results:
   - Run Summary
   - Assignment by Subject
   - Subjects by Group
   - Best fitness
   - Raw Output

7. If needed, click:
   Download assignment CSV

How to count covariates
When entering covariate counts:
- Count only the covariate columns
- Do not count the ID column

Example
If your file has:
- 1 ID column
- 3 quantitative covariates
- 2 categorical covariates

Then enter:
- Number of quantitative covariates = 3
- Number of categorical covariates = 2

Notes
- The number of subjects cannot be greater than the number of rows in the file
- The number of groups cannot be greater than the number of subjects
- At least one covariate must be provided
- The app assumes the first column is always the ID column

Common errors
1. File format not supported
Use only CSV, XLSX, or XLS.

2. Not enough columns
Make sure your file includes:
- 1 ID column
- the covariate columns you declared in the app

3. Invalid categorical variables
Categorical columns must contain only two possible values:
-1 and 1

Contact
If the app fails or your file structure is different, review the file format and covariate coding before running the algorithm.
