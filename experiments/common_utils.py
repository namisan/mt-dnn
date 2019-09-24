from data_utils import DataFormat


def dump_rows(rows, out_path):
    """
    output files should have following format
    :param rows:
    :param out_path:
    :return:
    """

    def detect_format(row):
        data_format = DataFormat.PremiseOnly
        if "hypothesis" in row:
            hypo = row["hypothesis"]
            if isinstance(hypo, str):
                data_format = DataFormat.PremiseAndOneHypothesis
            else:
                assert isinstance(hypo, list)
                data_format = DataFormat.PremiseAndMultiHypothesis
        return data_format

    with open(out_path, "w", encoding="utf-8") as out_f:
        row0 = rows[0]
        data_format = detect_format(row0)
        for row in rows:
            assert data_format == detect_format(row), row
            if data_format == DataFormat.PremiseOnly:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"]))
            elif data_format == DataFormat.PremiseAndOneHypothesis:
                for col in ["uid", "label", "premise", "hypothesis"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"], row["hypothesis"]))
            elif data_format == DataFormat.PremiseAndMultiHypothesis:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                hypothesis = row["hypothesis"]
                for one_hypo in hypothesis:
                    if "\t" in str(one_hypo):
                        import pdb; pdb.set_trace()
                hypothesis = "\t".join(hypothesis)
                out_f.write("%s\t%s\t%s\t%s\t%s\n" % (row["uid"], row["ruid"], row["label"], row["premise"], hypothesis))
            else:
                raise ValueError(data_format)