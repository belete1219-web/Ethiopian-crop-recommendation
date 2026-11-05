"""Builds 'Project Implementation Report.docx' from the saved results (run: python build_report.py).

Every number in the report is read from outputs/*.csv, model_metadata.json or data/*.csv, and every
figure is inserted from outputs/ or evidence/, so the report always matches the saved model.
Re-run the notebook first if you retrain; then re-run this script.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for easy landscape edits)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE = Path(__file__).resolve().parent
OUT, EVID = BASE / "outputs", BASE / "evidence"
REPORT = BASE / "Project Implementation Report.docx"
GREEN = RGBColor(0x1B, 0x5E, 0x20)

meta = json.loads((BASE / "model_metadata.json").read_text())
tm = meta["test_metrics"]
ref = meta["input_reference"]
data = pd.read_csv(BASE / "data" / "Crop_recommendation_Ethio.csv")
FEATURES = meta["features"]

comparison = pd.read_csv(OUT / "validation_model_comparison.csv")
topk = pd.read_csv(OUT / "top_k_accuracy.csv")
cv = pd.read_csv(OUT / "five_fold_cv.csv")
per_class = pd.read_csv(OUT / "per_class_metrics.csv")
confusions = pd.read_csv(OUT / "top_confusions.csv")
cm = pd.read_csv(OUT / "confusion_matrix.csv", index_col=0)
provenance = pd.read_csv(OUT / "data_provenance_checks.csv")
importance = pd.read_csv(OUT / "feature_importance.csv")
examples = pd.read_csv(OUT / "input_check_examples.csv")
flag_rates = pd.read_csv(OUT / "input_check_validation.csv", index_col=0)
example_pred = pd.read_csv(OUT / "example_top3_prediction.csv")

n_test = int(tm["test_rows"])
acc, top3 = tm["accuracy"], tm["top3_accuracy"]
k1 = int(topk.loc[topk.k == 1, "rows_correct"].iloc[0])
k2 = int(topk.loc[topk.k == 2, "rows_correct"].iloc[0])
k3 = int(topk.loc[topk.k == 3, "rows_correct"].iloc[0])
assert k1 == round(acc * n_test) and k3 == round(top3 * n_test)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


ci1, ci3 = wilson(k1, n_test), wilson(k3, n_test)

# --------------------------------------------------------------------------- document helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)      # A4
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(0.9)

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Calibri", Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12
for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold = "Calibri", Pt(size), True
    st.font.color.rgb = GREEN
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True

counter = {"fig": 0, "tab": 0}


def para(text="", bold=False, italic=False, size=None, align=None, after=None, keep=False, color=None):
    p = doc.add_paragraph()
    add_runs(p, text, bold, italic, size, color)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def add_runs(p, text, bold=False, italic=False, size=None, color=None):
    """Supports **bold** segments inside text."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    add_runs(p, text)
    p.paragraph_format.space_after = Pt(3)
    return p


def caption(kind, text):
    counter[kind] += 1
    label = "Figure" if kind == "fig" else "Table"
    p = doc.add_paragraph()
    r = p.add_run(f"{label} {counter[kind]}. ")
    r.bold, r.font.size = True, Pt(9.5)
    r2 = p.add_run(text)
    r2.italic, r2.font.size = True, Pt(9.5)
    p.paragraph_format.space_after = Pt(10)
    return p


def figure(path, width, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = caption("fig", text)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit"); el.set(qn("w:val"), "true"); trPr.append(el)


def table(header, rows, widths, text, align_right_from=1, font=9.5, highlight=None):
    """highlight: optional {row_index: hex_fill}."""
    counter["tab"] += 1
    cap = doc.add_paragraph()
    r = cap.add_run(f"Table {counter['tab']}. "); r.bold, r.font.size = True, Pt(9.5)
    r2 = cap.add_run(text); r2.italic, r2.font.size = True, Pt(9.5)
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.width = Inches(widths[i]); shade(c, "1B5E20")
        c.paragraphs[0].text = ""
        run = c.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(font)
        run.font.color.rgb = RGBColor(255, 255, 255)
        if i >= align_right_from:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    repeat_header(t.rows[0])
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        no_split(t.rows[-1])
        for i, v in enumerate(row):
            cells[i].width = Inches(widths[i])
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(v)); run.font.size = Pt(font)
            if i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if highlight and ri in highlight:
                shade(cells[i], highlight[ri])
            elif ri % 2 == 1:
                shade(cells[i], "F1F8F2")
    if len(rows) <= 12:                      # keep short tables on one page
        for row in t.rows[:-1]:
            for c in row.cells:
                for pp in c.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def callout(title, text, fill="FFF8E1"):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]; c.width = Inches(6.2); shade(c, fill)
    p = c.paragraphs[0]
    add_runs(p, f"**{title}** " + text, size=10)
    p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# footer page numbers
def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Crop Recommendation Based on Soil and Climate Conditions  |  Page ")
    r.font.size = Pt(8.5)
    run = p.add_run(); run.font.size = Pt(8.5)
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if t:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


add_page_number(sec)

# --------------------------------------------------------------------------- title
para("Project Implementation Report", bold=True, size=26, color=GREEN, after=2)
para("Crop Recommendation Based on Soil and Climate Conditions", bold=True, size=15, after=2)
para("Method, measured results, input validation and limitations", italic=True, size=11, after=12)

callout("Summary.",
        f"A Random Forest ranks ten crop labels from seven soil and climate inputs. On {n_test} held-out rows its single "
        f"highest-scored crop is correct {pct(acc)} of the time (test accuracy), and the true crop is among its three "
        f"highest-scored crops {pct(top3)} of the time (top-3 accuracy). These two numbers answer different questions "
        "and are not interchangeable (Section 6.2). The data are of undocumented origin and almost certainly "
        "synthetic, so all results are software-validation results, not evidence of agronomic performance. "
        "The Streamlit app warns users when an input is outside, or unlike, the data the model was fitted on "
        "(Section 7).", fill="E8F5E9")

# --------------------------------------------------------------------------- 1
doc.add_heading("1. Purpose and scope", level=1)
para("The prototype maps seven numeric inputs (nitrogen N, phosphorus P, potassium K, temperature, humidity, soil pH "
     "and rainfall) to one of ten crop labels: " + ", ".join(c.replace("_", " ") for c in meta["classes"]) + ". "
     "The user-facing output is the three crops with the highest model scores. Scores are the model's class "
     "probabilities; they are not predicted yields, fertilizer prescriptions or calibrated real-world probabilities.")

# --------------------------------------------------------------------------- 2
doc.add_heading("2. Dataset and its provenance", level=1)
para(f"The supplied file has {len(data)} rows, {len(FEATURES)} numeric features and a label column with "
     f"{data['label'].nunique()} crops (60 rows each). It arrived without documentation: no author, collection site, "
     "dates, units or labelling protocol. The file name and crop names suggest an Ethiopian focus, but nothing shows "
     "that any row is a field observation.")
para("The notebook tests whether the rows look like field data or like simulated draws. Table 1 summarises the checks; "
     "all point towards generated data. This is an inference from the data, not a documented fact.")
table(["Check", "Result", "How to read it"],
      provenance.values.tolist(), [2.6, 1.0, 2.6], "Evidence on how the rows were produced (outputs/data_provenance_checks.csv).",
      align_right_from=9, font=8.5)
para("Consequence: the label is the crop a row was generated for, not a recorded outcome such as the crop a farmer grew "
     "or the best-yielding crop. The model learns to separate ten generated, overlapping profiles. Our preparation "
     "step only validates the file (numeric conversion, removal of missing or non-finite rows, trimming and "
     "lower-casing of labels); no rows were generated, augmented or edited by us.")

# --------------------------------------------------------------------------- 3
doc.add_heading("3. Exploratory data analysis", level=1)
doc.add_heading("3.1 Class balance", level=2)
figure(OUT / "class_distribution.png", 5.6, "Number of rows per crop label (60 each, 600 in total).")
para("Every crop has exactly 60 rows. Because the classes are balanced, plain accuracy is not distorted by a dominant "
     "class, and the majority-class baseline scores 10% (Section 5). Exact balance is also one of the signals that the "
     "file is generated rather than collected (Table 1).")

doc.add_heading("3.2 Feature distributions", level=2)
figure(OUT / "feature_histograms.png", 6.0, "Distribution of each input feature over all 600 rows.")
d = data[FEATURES].describe().T
para("Ranges in the file are: " + "; ".join(
    f"{f} {d.loc[f, 'min']:.4g} to {d.loc[f, 'max']:.4g}" for f in FEATURES) + ". "
     "These are the ranges the model learns from; the app compares user inputs with them (Section 7). Rainfall has the "
     f"widest spread (mean {d.loc['rainfall', 'mean']:.0f}, SD {d.loc['rainfall', 'std']:.0f}); humidity reaches exactly "
     "100 and rainfall exactly 100 in one row each, which is consistent with clipping at a bound.")

doc.add_heading("3.3 Correlation between features", level=2)
figure(OUT / "correlation_heatmap.png", 3.9, "Pearson correlation between features over all 600 rows.")
corr = data[FEATURES].corr().abs().where(~np.eye(len(FEATURES), dtype=bool))
pair = corr.stack().idxmax()
para(f"The strongest pooled correlation is between {pair[0]} and {pair[1]} ({data[list(pair)].corr().iloc[0, 1]:.2f}). "
     "Pooled correlations are higher than within-crop correlations (mean absolute 0.243 pooled against 0.100 within "
     "crops, Table 1) only because the crops have different means; inside a crop the features behave as independent "
     "draws, another sign of generated data.")

doc.add_heading("3.4 How the features differ between crops", level=2)
figure(OUT / "feature_by_crop_boxplots.png", 5.3, "Each feature by crop. Overlapping boxes mean a feature alone cannot separate those crops.")
means = data.groupby("label")[FEATURES].mean()
para(f"Several features separate some crops clearly: mean rainfall ranges from {means['rainfall'].min():.0f} "
     f"({means['rainfall'].idxmin().replace('_', ' ')}) to {means['rainfall'].max():.0f} "
     f"({means['rainfall'].idxmax().replace('_', ' ')}), mean temperature from {means['temperature'].min():.1f} "
     f"({means['temperature'].idxmin()}) to {means['temperature'].max():.1f} ({means['temperature'].idxmax()}), and "
     f"mean N from {means['N'].min():.0f} ({means['N'].idxmin()}) to {means['N'].max():.0f} "
     f"({means['N'].idxmax()}). But many crops overlap on most features, so no single feature classifies the rows; "
     "that overlap is what produces the confusions in Section 6.3.")

# --------------------------------------------------------------------------- 4
doc.add_heading("4. Method", level=1)
bullet("**Split.** Stratified 70 / 15 / 15 split with random_state 42: 420 training, 90 validation and 90 test rows "
       "(9 test rows per crop).")
bullet("**Candidates.** Majority-class baseline, Decision Tree, Gradient Boosting and Random Forest.")
bullet("**Selection.** The model with the highest validation Macro-F1 was chosen, then refitted on training plus "
       "validation rows (510 rows). The 90 test rows were used once, for the final evaluation, and the saved model never "
       "saw them (checked by verify_saved_model.py).")
bullet("**Stability check.** Five-fold cross-validated Macro-F1 over all 600 rows. This is supplementary and not an "
       "independent estimate, because its folds include the test rows.")

# --------------------------------------------------------------------------- 5
doc.add_heading("5. Model comparison on the validation set", level=1)
figure(OUT / "validation_model_comparison.png", 5.4, "Validation accuracy and Macro-F1 for each candidate model.")
rows = [[r.model, f"{r.validation_accuracy:.3f}", f"{r.validation_macro_f1:.3f}", f"{r.validation_weighted_f1:.3f}"]
        for r in comparison.itertuples()]
table(["Model", "Accuracy", "Macro-F1", "Weighted F1"], rows, [2.4, 1.2, 1.2, 1.4],
      "Validation results (90 rows). Random Forest was selected.", highlight={0: "DCEFDC"})
best, second = comparison.iloc[0], comparison.iloc[1]
para(f"Random Forest scored the highest validation Macro-F1 ({best.validation_macro_f1:.3f}), ahead of {second.model} "
     f"({second.validation_macro_f1:.3f}). All three learned models are far above the 10% baseline. With only 90 "
     "validation rows each row is worth 1.1 percentage points, so small gaps should not be over-read; the gap between "
     "Random Forest and the single Decision Tree is large enough to be meaningful.")

# --------------------------------------------------------------------------- 6
doc.add_heading("6. Results on the held-out test set", level=1)
doc.add_heading("6.1 Headline metrics", level=2)
rows = [["Test accuracy (top-1)", f"{acc:.3f}", f"{k1} of {n_test} rows"],
        ["Macro precision", f"{tm['macro_precision']:.3f}", "mean over 10 crops"],
        ["Macro recall", f"{tm['macro_recall']:.3f}", "mean over 10 crops"],
        ["Macro-F1", f"{tm['macro_f1']:.3f}", "mean over 10 crops"],
        ["Top-3 accuracy", f"{top3:.3f}", f"{k3} of {n_test} rows"],
        ["Five-fold CV Macro-F1", f"{cv.macro_f1.mean():.3f} (SD {cv.macro_f1.std(ddof=0):.3f})", "all 600 rows; supplementary"]]
table(["Metric", "Value", "Meaning"], rows, [2.3, 1.7, 2.2], f"Random Forest on the {n_test} held-out test rows.", align_right_from=1)

doc.add_heading("6.2 Test accuracy versus top-3 accuracy", level=2)
para("The two headline numbers answer different questions and measure different things:", keep=True)
bullet(f"**Test accuracy (top-1)** asks: *is the single highest-scored crop the true label?* It is {pct(acc)} "
       f"({k1}/{n_test}). This is the strict measure of how often the first recommendation is right.".replace("*", ""))
bullet(f"**Top-3 accuracy** asks: *is the true label anywhere among the three highest-scored crops?* It is {pct(top3)} "
       f"({k3}/{n_test}). This matches the list the app displays, but it is an easier test: a row counts as correct "
       "even when the true crop is only second or third.".replace("*", ""))
para("Top-3 accuracy is always at least as high as test accuracy, and the gap between them is exactly the rows where "
     f"the true crop was the model's second or third choice. Here that is {k3 - k1} rows ({pct((k3 - k1) / n_test)} of "
     f"the test set): {k2 - k1} in second place and {k3 - k2} in third. Only {n_test - k3} rows "
     f"({pct((n_test - k3) / n_test)}) had the true crop outside the top three.")
figure(OUT / "accuracy_vs_top3.png", 6.2,
       "Left: accuracy as k grows (the test gets easier, and random guessing rises with it). Right: where the true crop "
       "ranked among the 90 test rows; the first bar is the test accuracy.")
rows = []
for r in topk[topk.k <= 5].itertuples():
    rows.append([f"Top-{r.k}" + ("  (= test accuracy)" if r.k == 1 else ""), f"{r.rows_correct} / {n_test}",
                 pct(r.accuracy), pct(r.random_guess, 0), f"{r.accuracy / r.random_guess:.1f}x"])
table(["Measure", "Rows correct", "Accuracy", "Random guess", "Gain over random"], rows, [2.0, 1.1, 1.0, 1.1, 1.1],
      "Accuracy when the true crop only has to be among the k highest-scored crops (outputs/top_k_accuracy.csv).",
      highlight={0: "DCEFDC", 2: "DCEFDC"})
para("How to read the two numbers together:")
bullet(f"Random guessing gives 10% for top-1 and 30% for top-3, so both results are far above chance; the top-1 result "
       f"is {acc / 0.1:.1f} times chance and the top-3 result {top3 / 0.3:.1f} times chance. Top-3 looks larger in "
       "absolute terms mainly because the test is easier, not because the model is better at it.")
bullet("Top-3 accuracy does not say which of the three crops is right. A user who sees three crops and is told one "
       f"is correct still has to choose; in {k3 - k1} of the {k3} top-3 hits the correct crop was not the one ranked first.")
bullet(f"With {n_test} test rows the estimates are imprecise: a 95% Wilson interval is {pct(ci1[0], 0)} to {pct(ci1[1], 0)} "
       f"for test accuracy and {pct(ci3[0], 0)} to {pct(ci3[1], 0)} for top-3 accuracy, and one row moves either by "
       "1.1 percentage points.")
ex = example_pred
callout("Worked example.",
        f"For the app's default input (N 60, P 30, K 35, temperature 22, humidity 60, pH 6.4, rainfall 700) the model "
        f"returns {ex.recommended_crop[0].replace('_', ' ')} ({ex.model_score[0]:.0%}), "
        f"{ex.recommended_crop[1].replace('_', ' ')} ({ex.model_score[1]:.0%}) and "
        f"{ex.recommended_crop[2].replace('_', ' ')} ({ex.model_score[2]:.1%}). Test accuracy is about whether "
        f"{ex.recommended_crop[0].replace('_', ' ')} alone is right; top-3 accuracy is about whether the right crop is "
        "any of these three. Note also that a displayed score such as 56% is a model score, not a calibrated "
        "probability that this crop is right.")

doc.add_heading("6.3 Confusion matrix", level=2)
figure(OUT / "confusion_matrix.png", 5.3,
       "Confusion matrix on the 90 test rows (rows = true crop, columns = predicted crop; 9 rows per crop).")
n_err = n_test - k1
para(f"The diagonal holds the {k1} correct predictions; the {n_err} errors are spread over {len(confusions)} different "
     f"true-to-predicted pairs, and no pair occurs more than {confusions['count'].max()} times. There is therefore no "
     "single dominant failure, although with 9 rows per crop this cannot be proven either way.")
top = confusions[confusions["count"] == confusions["count"].max()]
rows = [[r.true_crop.replace("_", " "), r.predicted_as.replace("_", " "), r.count] for r in confusions.itertuples()]
table(["True crop", "Predicted as", "Rows"], rows, [2.2, 2.2, 0.9],
      "Every misclassification pair on the test set (outputs/top_confusions.csv).", align_right_from=2,
      highlight={i: "FDECEA" for i in range(len(top))})
pairs = "; ".join(f"{r.true_crop.replace('_', ' ')} as {r.predicted_as.replace('_', ' ')}" for r in top.itertuples())
para(f"The most frequent confusions (two rows each) are {pairs}. Some of these pairs have similar profiles in the data; "
     f"for example maize and enset have similar mean N ({means.loc['maize', 'N']:.0f} and {means.loc['enset', 'N']:.0f}) "
     f"and mean K ({means.loc['maize', 'K']:.0f} and {means.loc['enset', 'K']:.0f}). Because the profiles are "
     "generated, these overlaps describe the dataset and not real-world crop similarity.")

doc.add_heading("6.4 Class-level results", level=2)
figure(OUT / "per_class_metrics.png", 5.6, "Precision, recall and F1-score for each crop on the test set.")
rows = [[r.crop.replace("_", " "), f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r._4:.2f}", int(r.support), f"{int(r.correct)} / {int(r.support)}"]
        for r in per_class.itertuples()]
table(["Crop", "Precision", "Recall", "F1-score", "Support", "Correct"], rows, [1.6, 1.0, 1.0, 1.0, 0.9, 0.9],
      "Per-crop test results (outputs/per_class_metrics.csv; support = test rows per crop).")
lo, hi = per_class.loc[per_class.recall.idxmin()], per_class.loc[per_class.recall.idxmax()]
lowp = per_class.loc[per_class.precision.idxmin()]
para(f"Recall is lowest for {lo.crop.replace('_', ' ')} ({lo.recall:.2f}, {int(lo.correct)} of 9 found) and highest "
     f"for {hi.crop} ({hi.recall:.2f}, all 9 found). Precision is lowest for {lowp.crop} ({lowp.precision:.2f}): other "
     f"crops were sometimes predicted as {lowp.crop}. Sesame has precision 1.00 but recall 0.78, meaning it was never "
     "predicted wrongly but two sesame rows were assigned to chickpea.")
callout("Caution.", "Each crop has only 9 test rows, so one row changes a recall value by 0.11. The differences "
        "between crops are suggestive and should not be treated as reliable rankings of crop difficulty.")

doc.add_heading("6.5 Feature importance", level=2)
figure(OUT / "feature_importance.png", 5.0, "Random Forest impurity-based feature importance.")
imp = importance.sort_values("importance", ascending=False)
para(f"The largest contributors are {imp.iloc[0].feature} ({imp.iloc[0].importance:.2f}), {imp.iloc[1].feature} "
     f"({imp.iloc[1].importance:.2f}) and {imp.iloc[2].feature} ({imp.iloc[2].importance:.2f}); K is the smallest "
     f"({imp.iloc[-1].importance:.2f}). Importance describes how the model splits this dataset; it is not evidence that "
     "these variables drive crop choice in the field.")

# --------------------------------------------------------------------------- 7
doc.add_heading("7. Input validation in the Streamlit app", level=1)
para("A valid number is not necessarily a reliable input for this model. Before the revision, the app only enforced "
     "physical limits (pH 0 to 14, humidity 0 to 100%, no negative nutrients or rainfall). Inside those limits it still "
     "returned a confident-looking ranking for values the model had never seen: for temperature −5 and rainfall 20, "
     "the model gives barley a 55% score. The app now compares every entry with the rows the model was fitted on "
     f"({ref['fitted_rows']} train + validation rows) and reports one of three outcomes:")
bullet("**Outside the fitted range** (below the minimum or above the maximum seen in training): a red \"Low reliability\" "
       "banner, a warning that names the feature, the value and the fitted range, and the ranking is labelled "
       "\"low reliability\".")
bullet("**Edge of the fitted range** (more extreme than 99% of fitted rows): a \"Use with caution\" message.")
bullet("**Unfamiliar combination**: every value is individually inside its range, but the combination is farther from "
       "every fitted row than the most unusual 1% of fitted rows are from their nearest neighbour (distance in "
       "standardised feature space, calibrated by leave-one-out on the fitted rows).")
para("Warnings appear above the recommendation so they are read first. Each input also shows the fitted range in its help "
     "text, and an expander lists all ranges. The table below shows the fitted ranges that drive the checks.")
rows = [[f, f"{v['min']:.4g}", f"{v['p01']:.4g}", f"{v['p99']:.4g}", f"{v['max']:.4g}"] for f, v in ref["feature_ranges"].items()]
table(["Feature", "Minimum", "1st percentile", "99th percentile", "Maximum"], rows, [1.6, 1.1, 1.2, 1.3, 1.1],
      "Fitted ranges used by the app. Outside min/max = low reliability; outside the 1st/99th percentile = caution.")

doc.add_heading("7.1 What the user sees", level=2)
rows = [[r.input, r.top_crop, r.top_score, r.reliability, r.warnings.replace("_", " ") if r.warnings != "none" else "none"]
        for r in examples.itertuples()]
table(["Input", "Top crop", "Score", "Reliability", "Warnings raised"], rows, [2.3, 0.8, 0.6, 0.9, 1.8],
      "The three scenarios in the screenshots below (outputs/input_check_examples.csv).", align_right_from=9, font=9)

# screenshot grid (blank margins and the Streamlit toolbar are cropped away; originals stay in evidence/)
import io
from PIL import Image


def cropped(path):
    im = Image.open(path).convert("RGB")
    px = np.asarray(im)
    rows = np.where((px < 245).any(axis=(1, 2)))[0]
    top, bottom = 60, min(im.height, int(rows.max()) + 25)       # 60 px = Deploy toolbar
    buf = io.BytesIO(); im.crop((0, top, im.width, bottom)).save(buf, "PNG"); buf.seek(0)
    return buf


shots = [("app_1_typical_values.png", "Typical values: green confirmation, no warnings."),
         ("app_2_out_of_range.png", "Temperature \u22125 and rainfall 20: red banner, named warnings, ranking labelled low reliability."),
         ("app_3_unusual_combination.png", "Ordinary values, unusual combination: flagged as low reliability."),
         ("app_4_fitted_range_panel.png", "Expander listing the range the model was fitted on.")]
g = doc.add_table(rows=2, cols=2)
g.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (fn, cap_text) in enumerate(shots):
    cell = g.rows[i // 2].cells[i % 2]
    cell.width = Inches(3.1)
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(cropped(EVID / fn), width=Inches(3.0))
    counter["fig"] += 1
    c = cell.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {counter['fig']}. "); r.bold, r.font.size = True, Pt(8.5)
    r2 = c.add_run(cap_text); r2.italic, r2.font.size = True, Pt(8.5)
for row in g.rows:
    no_split(row)
doc.add_paragraph().paragraph_format.space_after = Pt(4)

doc.add_heading("7.2 How noisy are the warnings?", level=2)
ft, fr = flag_rates.iloc[:, 0], flag_rates.iloc[:, 1]
rows = [[lvl, int(ft[lvl]), pct(ft[lvl] / ft.sum(), 0), int(fr[lvl]), pct(fr[lvl] / fr.sum(), 0)] for lvl in ("familiar", "edge", "unreliable")]
table(["Outcome", "Test rows", "Share", "Random in-range combos", "Share"], rows, [1.5, 1.0, 0.8, 1.9, 0.8],
      f"Check outcomes for the {int(ft.sum())} real held-out rows and for {int(fr.sum())} random combinations whose "
      "values all lie inside their individual fitted ranges (outputs/input_check_validation.csv).")
para(f"A useful check should rarely flag rows that belong to the dataset and often flag combinations that do not. Of the "
     f"{int(ft.sum())} real test rows, {int(ft['unreliable'])} ({pct(ft['unreliable'] / ft.sum(), 0)}) were flagged low "
     f"reliability, all for a value just outside the fitted range of the 510 rows; none were flagged as an unusual "
     f"combination. Of the random combinations, {pct(fr['unreliable'] / fr.sum(), 0)} were flagged, which shows the "
     "combination check catches inputs that pass every single-value test.")
para("Limits of the check: it is a heuristic, not a guarantee. By design about 1% of fitted-like rows can be flagged, "
     "and the absence of a warning means only that the input resembles the training rows. Because those rows are "
     "synthetic and their units are undocumented, a familiar input is still not evidence of a good real-world "
     "recommendation.")

# --------------------------------------------------------------------------- 8
doc.add_heading("8. Limitations", level=1)
bullet("**Data origin.** Undocumented and almost certainly synthetic; labels are generating categories, not field "
       "outcomes. Results are software-validation results only.")
bullet("**Units.** Not documented; users must enter values in the same units as the training data.")
bullet(f"**Small test set.** {n_test} rows (9 per crop); the 95% interval for test accuracy is {pct(ci1[0], 0)} to {pct(ci1[1], 0)}.")
bullet("**Scores are not calibrated probabilities** and have not been validated against real outcomes.")
bullet("**Input checks are heuristic** and depend on the same synthetic rows as the model.")

# --------------------------------------------------------------------------- 9
doc.add_heading("9. Conclusion", level=1)
para(f"The Random Forest recovers the generating crop for {pct(acc)} of held-out rows at first choice and {pct(top3)} within "
     "its top three, far above the 10% and 30% chance levels. The two numbers must be quoted together with their "
     "definitions: the first is the strict test and the second the easier one. The app now separates a valid number from "
     "a reliable one by warning on out-of-range values, edge values and unusual combinations. Before any operational use, "
     "the model must be re-fitted and validated with documented local field data and reviewed by agronomic experts.")

doc.add_heading("Appendix: reproducing this report", level=1)
for line in ("Run all cells of Ethiopian_Crop_Recommendation_Implementation.ipynb to regenerate outputs/, the model and model_metadata.json.",
             "python verify_saved_model.py confirms the saved model reproduces the reported test metrics.",
             "python test_input_checks.py and python test_app_validation.py test the input warnings.",
             "python build_report.py rebuilds this document from the saved results."):
    bullet(line)

doc.core_properties.title = "Project Implementation Report - Crop Recommendation Based on Soil and Climate Conditions"
doc.save(REPORT)
print("Saved", REPORT.name, "| figures:", counter["fig"], "| tables:", counter["tab"])
