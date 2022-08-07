#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = pd.read_csv("data/original/Graduation Rate Database/GRAD_RATE_AND_OUTCOMES_2020.csv", low_memory=False)


#%%
# drop out rates and other rates are rounder to whole percentage points in the original data
# the below code makes them calculated more precisely
data[['enroll_cnt', 'grad_cnt', 'non_diploma_credential_cnt', 'still_enr_cnt', 'dropout_cnt']] = \
    data[['enroll_cnt', 'grad_cnt', 'non_diploma_credential_cnt', 'still_enr_cnt', 'dropout_cnt']]\
    .query('dropout_cnt != "-"').apply(pd.to_numeric)
data = data.eval('dropout_pct = dropout_cnt / enroll_cnt')

#%%
# Unfortunately, it was not possible to extract the data in a single database
# so in order not to spend too much time with downloading the data for each county separately,
# I had to find patterns in url and download them directly in Python using the loop.
# Names of almost all counties in the URLs were the same as in NYSED database, except for New York and Saint Lawrence
# So I had to temporarily rename these counties.

# PLEASE ALLOW SEVERAL MINUTES FOR THIS PIECE OF CODE TO RUN!

NY_counties = list(set(data.query('aggregation_type == "County"').county_name))
NY_counties[NY_counties.index('SAINT LAWRENCE')] = "STLAWRENCE"  # to align the name with census.gov
NY_counties[NY_counties.index('NEW YORK')] = "NEWYORK"  # to align the name with census.gov

county_urls = []
for i in [6 * i for i in range(11)]:
    if i < 60:
        county_urls.append(
            f"https://www.census.gov/quickfacts/fact/csv/{NY_counties[i]}countynewyork,"
            f"{NY_counties[i + 1]}countynewyork,{NY_counties[i + 2]}countynewyork,"
            f"{NY_counties[i + 3]}countynewyork,{NY_counties[i + 4]}countynewyork,"
            f"{NY_counties[i + 5]}countynewyork,PST045219")
    else:
        county_urls.append(
            f"https://www.census.gov/quickfacts/fact/csv/{NY_counties[i]}countynewyork,"
            f"{NY_counties[i + 1]}countynewyork,PST045219")

county_data = pd.read_csv(county_urls[0])

for i in range(1, 11):
    county_data = county_data.merge(pd.read_csv(county_urls[i]))

county_data = county_data.iloc[:62] # remove empty columns and keep data for 62 counties in New York state
county_data = county_data[county_data.columns.drop(list(county_data.filter(regex='Note')))]

county_data = county_data.set_index("Fact")
county_data = county_data.T
county_data = county_data.reset_index()
county_data = county_data.rename(columns={"index": "county_name"})


#%%
# back to the names used in NYSED databases
county_data['county_name'] = county_data['county_name'].replace(regex=' County, New York', value="")
county_data['county_name'] = county_data['county_name'].replace(regex='St. Lawrence', value="Saint Lawrence")
county_data['county_name'] = county_data['county_name'].str.upper()


#%%
county_data = county_data\
    .drop(columns=['Population Estimates, July 1 2021, (V2021)',
                   'Population estimates base, April 1, 2020, (V2021)',
                   'Population, percent change - April 1, 2020 (estimates base) to July 1, 2021, (V2021)',
                   'Population, Census, April 1, 2010', 'Persons under 5 years, percent', 'White alone, percent',
                   'Black or African American alone, percent', 'American Indian and Alaska Native alone, percent',
                   'Asian alone, percent', 'Native Hawaiian and Other Pacific Islander alone, percent',
                   'Two or More Races, percent',
                   'Hispanic or Latino, percent', 'White alone, not Hispanic or Latino, percent',
                   'Veterans, 2016-2020', 'Housing units,  July 1, 2019,  (V2019)',
                   'Median selected monthly owner costs -with a mortgage, 2016-2020',
                   'Median selected monthly owner costs -without a mortgage, 2016-2020',
                   'Median gross rent, 2016-2020', 'Building permits, 2020',
                   'Households, 2016-2020',
                   'In civilian labor force, total, percent of population age 16 years+, 2016-2020',
                   'In civilian labor force, female, percent of population age 16 years+, 2016-2020',
                   'Total accommodation and food services sales, 2012 ($1,000)',
                   'Total health care and social assistance receipts/revenue, 2012 ($1,000)',
                   'Total manufacturers shipments, 2012 ($1,000)',
                   'Total retail sales, 2012 ($1,000)', 'Total retail sales per capita, 2012',
                   'Per capita income in past 12 months (in 2020 dollars), 2016-2020',
                   'Total employer establishments, 2020', 'Total employment, 2020',
                   'Total annual payroll, 2020 ($1,000)', 'Total employment, percent change, 2019-2020',
                   'Total nonemployer establishments, 2018', 'All firms, 2012', 'Men-owned firms, 2012',
                   'Women-owned firms, 2012', 'Minority-owned firms, 2012', 'Nonminority-owned firms, 2012',
                   'With a disability, under age 65 years, percent, 2016-2020',
                   'Persons  without health insurance, under age 65 years, percent',
                   'Veteran-owned firms, 2012', 'Nonveteran-owned firms, 2012',
                   'Population per square mile, 2010', 'FIPS Code'])



#%%
county_data[['Persons per household, 2016-2020',
             'Mean travel time to work (minutes), workers age 16 years+, 2016-2020']] = \
    county_data[['Persons per household, 2016-2020',
                 'Mean travel time to work (minutes), workers age 16 years+, 2016-2020']].apply(pd.to_numeric)

#%%
county_data[['Median value of owner-occupied housing units, 2016-2020', 'Median household income (in 2020 dollars), 2016-2020',
             'Land area in square miles, 2010', 'Population, Census, April 1, 2020']] = \
    county_data[['Median value of owner-occupied housing units, 2016-2020', 'Median household income (in 2020 dollars), 2016-2020',
                 'Land area in square miles, 2010', 'Population, Census, April 1, 2020']]\
    .replace(regex='[$,]', value="").apply(pd.to_numeric)

#%%
# transforming percentage columns in numeric format
county_percentages_columns = []

for column in county_data.columns:
    try:
        if county_data[column].str.contains("%").sum() != 0:
            county_percentages_columns.append(column)
        else:
            pass
    except AttributeError:
        pass

county_data[county_percentages_columns] = county_data[county_percentages_columns]\
    .replace(regex='%', value="").apply(pd.to_numeric)/100


#%%
# finally calculating county population density and dropping county population and area columns
county_data = county_data.rename(columns={"Population, Census, April 1, 2020": "county_pop",
                                          "Land area in square miles, 2010": "county_area"})\
    .eval("cty_pop_density = county_pop / county_area")\
    .drop(columns=['county_pop', 'county_area'])

#%%
# to rename all remaining variables
county_data = county_data.rename(columns={
    'Persons under 18 years, percent': 'cty_pop_under_18',
    'Persons 65 years and over, percent': 'cty_pop_over_65',
    'Female persons, percent': 'cty_female',
    'Foreign born persons, percent, 2016-2020': 'cty_foreign_born',
    'Owner-occupied housing unit rate, 2016-2020': 'cty_owner_occupied',
    'Median value of owner-occupied housing units, 2016-2020': 'cty_housing_unit_median_value',
    'Persons per household, 2016-2020': 'cty_pers_pers_household',
    'Living in same house 1 year ago, percent of persons age 1 year+, 2016-2020': 'cty_same_house',
    'Language other than English spoken at home, percent of persons age 5 years+, 2016-2020': 'cty_non_english',
    'Households with a computer, percent, 2016-2020': 'cty_with_computer',
    'Households with a broadband Internet subscription, percent, 2016-2020': 'cty_with_internet',
    'High school graduate or higher, percent of persons age 25 years+, 2016-2020': 'cty_hs_graduates',
    "Bachelor's degree or higher, percent of persons age 25 years+, 2016-2020": 'cty_uni_degree',
    'Mean travel time to work (minutes), workers age 16 years+, 2016-2020': 'cty_time_to_work',
    'Median household income (in 2020 dollars), 2016-2020': 'cty_med_household_income',
    'Persons in poverty, percent': 'cty_poverty'})

#%%
# save processed county data
county_data.to_csv('data/processed/county_data.csv')

#%%
# load processed county data
county_data = pd.read_csv('data/processed/county_data.csv', index_col=0)

#%%
# information where there are fewer than 5 students in the
# group is suppressed for confidentiality reasons
# and all data is indicated as "-" there, so it removed it from the model.
data_combined = data.query('aggregation_index == 4').query('subgroup_name == ["Female","Male"]')\
    .query('grad_pct != "-"')\
    .query('membership_desc != "2014 Total Cohort - 6 Year Outcome - August 2020"')\
    .query('membership_desc != "2015 Total Cohort - 5 Year Outcome - August 2020"')\
    .query('membership_desc != "2016 Total Cohort - 4 Year Outcome - August 2020"')\
    .drop(columns=['report_school_year', 'aggregation_index', 'aggregation_type', 'entity_inactive_date', 'lea_beds',
                   'lea_name', 'nyc_ind', 'nrc_code', 'boces_code', 'boces_name', 'membership_code', 'membership_key',
                   'subgroup_code', 'grad_cnt', 'grad_pct', 'local_cnt', 'local_pct', 'reg_cnt',
                   'reg_pct', 'reg_adv_cnt', 'reg_adv_pct', 'non_diploma_credential_cnt', 'still_enr_cnt', 'ged_cnt',
                   'ged_pct', 'county_code', 'non_diploma_credential_pct', 'enroll_cnt', 'dropout_cnt',
                   'still_enr_pct', 'nrc_desc'])\
    .rename(columns={"subgroup_name": "gender", "membership_desc": "cohort"})


#%%
data_teachers_pupils = pd.merge(
    left=pd.read_csv('data/original/Report Card Database/Expenditures per Pupil.csv').query('YEAR == 2020'),
    right=pd.read_csv('data/original/Report Card Database/Inexperienced Teachers and Principals.csv').query('YEAR == 2020'),
    how='inner',
    on='ENTITY_CD')[['ENTITY_CD', 'PUPIL_COUNT_TOT', 'PER_FEDERAL_EXP',
                     'PER_STATE_LOCAL_EXP', 'NUM_TEACH', 'PER_TEACH_INEXP']]


#%%
data_teachers_pupils = data_teachers_pupils.eval("TEACH_PER_PUPIL = NUM_TEACH / PUPIL_COUNT_TOT")\
    .drop(columns='NUM_TEACH')


#%%
schools = data_combined['aggregation_code'].unique()
data_attendance = pd.read_csv('data/original/Student and Educator Database/Attendance.csv')\
    .query('YEAR == 2020').query('ENTITY_CD in @schools').eval('ATTENDANCE_RATE = ATTENDANCE_RATE / 100')\
    [['ENTITY_CD', 'ATTENDANCE_RATE']]

data_staff = pd.read_csv('data/original/Student and Educator Database/Staff.csv')\
    .query('YEAR == 2020').query('ENTITY_CD in @schools')[['ENTITY_CD', 'NUM_COUNSELORS', 'NUM_SOCIAL']]

data_suspensions = pd.read_csv('data/original/Student and Educator Database/Suspensions.csv')\
    .query('YEAR == 2020').query('ENTITY_CD in @schools').eval('PER_SUSPENSIONS = PER_SUSPENSIONS / 100')\
    [['ENTITY_CD', 'PER_SUSPENSIONS']]

data_lunch = pd.read_csv('data/original/Student and Educator Database/Free Reduced Price Lunch.csv')\
    .query('YEAR == 2020').query('ENTITY_CD in @schools').eval('PER_FREE_LUNCH = PER_FREE_LUNCH / 100')\
    [['ENTITY_CD', 'PER_FREE_LUNCH']]


#%%
data_combined = data_combined.merge(data_teachers_pupils, how='left', left_on='aggregation_code', right_on='ENTITY_CD')\
    .drop(columns='ENTITY_CD')
data_combined = data_combined.merge(data_attendance, how='left', left_on='aggregation_code', right_on='ENTITY_CD')\
    .drop(columns='ENTITY_CD')
data_combined = data_combined.merge(data_staff, how='left', left_on='aggregation_code', right_on='ENTITY_CD')\
    .drop(columns='ENTITY_CD')
data_combined = data_combined.merge(data_suspensions, how='left', left_on='aggregation_code', right_on='ENTITY_CD')\
    .drop(columns='ENTITY_CD')
data_combined = data_combined.merge(data_lunch, how='left', left_on='aggregation_code', right_on='ENTITY_CD')\
    .drop(columns='ENTITY_CD')
data_combined = data_combined.merge(county_data, how='left', on='county_name')
data_combined.index = data_combined.index.rename('ind')

#%%
# to correct remaining variables containing "%" symbol and divide by 100 percent of inexperienced teachers and principals
data_combined['PER_TEACH_INEXP'] = data_combined['PER_TEACH_INEXP'].replace(regex='%', value="").apply(pd.to_numeric)/100

#%%
data_combined = data_combined.eval("COUNSELORS_PER_PUPIL = NUM_COUNSELORS / PUPIL_COUNT_TOT")\
    .eval('SOCIAL_PER_PUPIL = NUM_SOCIAL / PUPIL_COUNT_TOT').drop(columns=['NUM_COUNSELORS', 'NUM_SOCIAL'])


#%%
# county_name is set as index as name of the county is not important from the model and was used to
# extract county related features (starting with "cty_").
# it should be noted that "cohort" variable is kept in the model as each cohort has different curriculum
# (of different duration) and therefore considered important to keep for further analysis.
data_combined = data_combined.set_index(['aggregation_code', 'county_name'], append=True)

#%%
# saving combined data
data_combined.to_csv('data/processed/data_combined.csv')

#%%
# loading combined data in case to skip previous steps
data_combined = pd.read_csv('data/processed/data_combined.csv', index_col=['ind', 'aggregation_code', 'county_name'])

#%%
# Dealing with missing values
missing_data = data_combined[data_combined.isnull().any(axis=1)]

missing_data.groupby('aggregation_code').agg('count').shape[0]
data_combined.groupby('aggregation_code').agg('count').shape[0]

missing_data.shape[0]
data_combined.shape[0]

#%%
# 3.07% of schools that have 1.48% of all pupils
print(round(missing_data.groupby('aggregation_name').agg('count').shape[0] / \
data_combined.groupby('aggregation_name').agg('count').shape[0]*100, 2), "%")

print(round(missing_data.groupby('aggregation_name').agg('mean')['PUPIL_COUNT_TOT'].sum() / \
data_combined.groupby('aggregation_name').agg('mean')['PUPIL_COUNT_TOT'].sum()*100, 2), "%")


#%%
# distribution of dropout rate in missing data
sns.histplot(data=data_combined, x='dropout_pct', kde=True)
plt.show()
sns.histplot(data=missing_data, x='dropout_pct', kde=True)
plt.show()

#%%
# dropping missing data from the dataset
data_combined = data_combined.dropna()
