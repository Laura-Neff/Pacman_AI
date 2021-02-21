
countries = ["Iran", "United States of America", "Australia", "China", "Burma", "India", "Peru", "United Arab Emirates"]
#Lists use brackets

lowerList = [country.lower() for country in countries if len(country) > 5];
print(lowerList)