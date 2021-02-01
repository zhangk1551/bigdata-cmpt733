# similarity_join.py
import re
import pandas as pd


class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        """
        Input: $df represents a DataFrame
               $cols represents the list of columns (in $df) that will be concatenated and be tokenized

        Output: Return a new DataFrame that adds the "joinKey" column to the input $df

        Comments: The "joinKey" column is a list of tokens, which is generated as follows:
                 (1) concatenate the $cols in $df;
                 (2) apply the tokenizer to the concatenated string
        Here is how the tokenizer should work:
                 (1) Use "re.split(r'\W+', string)" to split a string into a set of tokens
                 (2) Convert each token to its lower-case
        """
        df["joinKey"] = df[cols].apply(
            lambda x: re.split(r"\W+", " ".join([str(item) for item in x]).lower()),
            axis=1,
        )
        df["joinKey"] = df["joinKey"].apply(
            lambda keys: [key for key in keys if str(key) != "nan"]
        )
        return df

    def filtering(self, df1, df2):
        """
        Input: $df1 and $df2 are two input DataFrames, where each of them
               has a 'joinKey' column added by the preprocess_df function

        Output: Return a new DataFrame $cand_df with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
                where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
                Intuitively, $cand_df is the joined result between $df1 and $df2 on the condition that
                their joinKeys share at least one token.

        Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons,
                  you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function.
                  Please come up with a more efficient algorithm (see hints in Lecture 2).
        """

        def calc_token_index(df):
            df_token_index = {}
            for index, row in df.iterrows():
                for token in row["joinKey"]:
                    if df_token_index.get(token):
                        df_token_index[token].append(row["id"])
                    else:
                        df_token_index[token] = [row["id"]]
            return df_token_index

        df1_token_index = calc_token_index(df1)
        df2_token_index = calc_token_index(df2)
        df1 = df1.set_index("id")
        df2 = df2.set_index("id")
        cand_df = pd.DataFrame(columns=["id1", "joinKey1", "id2", "joinKey2"])
        for df1_token in df1_token_index.keys():
            if df2_token_index.get(df1_token):
                for df1_index in df1_token_index[df1_token]:
                    for df2_index in df2_token_index[df1_token]:
                        cand_df = cand_df.append(
                            {
                                "id1": df1_index,
                                "joinKey1": df1.loc[df1_index]["joinKey"],
                                "id2": df2_index,
                                "joinKey2": df2.loc[df2_index]["joinKey"],
                            },
                            ignore_index=True,
                        )
        return cand_df.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)

    def verification(self, cand_df, threshold):
        """
                    Input: $cand_df is the output DataFrame from the 'filtering' function.
                           $threshold is a float value between (0, 1]
        ​
                    Output: Return a new DataFrame $result_df that represents the ER result.
                            It has five columns: id1, joinKey1, id2, joinKey2, jaccard
        ​
                    Comments: There are two differences between $cand_df and $result_df
                              (1) $result_df adds a new column, called jaccard, which stores the jaccard similarity
                                  between $joinKey1 and $joinKey2
                              (2) $result_df removes the rows whose jaccard similarity is smaller than $threshold
        """
        result_df = cand_df
        result_df["jaccard"] = result_df.apply(
            lambda row: len(set(row["joinKey1"]).intersection(row["joinKey2"]))
            / len(set(row["joinKey1"]).union(row["joinKey2"])),
            axis=1,
        )
        result_df = result_df[result_df["jaccard"] >= threshold].reset_index(drop=True)
        return result_df

    def evaluate(self, result, ground_truth):
        """
        Input: $result is a list of matching pairs identified by the ER algorithm
               $ground_truth is a list of matching pairs labeld by humans

        Output: Compute precision, recall, and fmeasure of $result based on $ground_truth, and
                return the evaluation result as a triple: (precision, recall, fmeasure)

        """
        correct = sum([1 for x1 in result for x2 in ground_truth if x1 == x2])
        precision = correct / len(result)
        recall = correct / len(ground_truth)
        fmeasure = 2 * precision * recall / (precision + recall)
        return precision, recall, fmeasure

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print(
            "Before filtering: %d pairs in total"
            % (self.df1.shape[0] * self.df2.shape[0])
        )

        cand_df = self.filtering(new_df1, new_df2)
        print("After Filtering: %d pairs left" % (cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print("After Verification: %d similar pairs" % (result_df.shape[0]))

        return result_df


if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[["id1", "id2"]].values.tolist()
    ground_truth = pd.read_csv(
        "Amazon_Google_perfectMapping_sample.csv"
    ).values.tolist()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
