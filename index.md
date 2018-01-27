## Bank Risk Model evaluation
Exercise in which we'll process a large dataset of customers records with over 130 features each. In the exercise a number of steps is done prior to launch and evaluate models:
- *Feature Review* --> revision of features that could present problems that invalidate them: high porportion of `NaN`values, invalid ones, etc. Also, in some cases transformations are to be applied to features: scale, cathegorical text to number format, dates, etc.
- *EDA* --> Exploratory Data Analysis. Further insights on data, to check distribution, correlations, etc. When strong correlation between features are found (except with target) some features can be discarded as well.
- *Model workout* --> Model parametrization and training workout.

At this point, a number of remakrs have to be done. The first one, is that this binary classification problem (customer is predicted to be paying back or not), but with highly unbalanced classes (96.5% customers pay, only 3.5% don't), what makes that model scoring has to take into account accuracy for both classes. Also, feature discrimination is important at this point, thus `RFECV` is used for that. Only linear models, in particular `LogisticRegression`, are used at this point, as well as feature scaling. All these combinations leave computationally heavy models, but results up to >99% accuracy in paid credit predictions and > 70% in unpaid ones are obtained.

### Datasets

Heavy datasets area used, should you want a copy let me know at jorge.rocha.blanco@gmail.com
