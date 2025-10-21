from ucimlrepo import fetch_ucirepo

census_income = fetch_ucirepo(id=20)

X = census_income.data.features
y = census_income.data.targets

print(census_income.metadata)

print(census_income.variables)
