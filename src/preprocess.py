def streamline_features(directory):
    #Import Pandas
    import pandas as pd

    #Load Data
    df = pd.read_csv(directory)
    df = df.fillna(0)

    #Clean Data
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent", "loan_grade"], drop_first=True)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})

    #Sort Data
    X = df.drop(columns=["loan_status"])
    Y = df["loan_status"]

    #Return Data
    return X, Y