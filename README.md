# Tutorial Cross Validation
This repository contains the code, data and images used in the blog post at https://www.mdelcueto.com/blog/a-brief-guide-to-cross-validation

---
## Contents
- **generate_data.py**: it generates and plots x<sub>1</sub>,x<sub>2</sub>,f(x<sub>1</sub>,x<sub>2</sub>) data
- **validation.py**: uses regular validation to calculate RMSE and r of KRR model of dataset
- **kfold.py**: uses k-fold cross-validation to calculate RMSE and r of KRR model of dataset
- **loo.py**: uses leave-one-out cross-validation to calculate RMSE and r of KRR model of dataset
- **figures**: folder with all figures used in article

---

## Prerequisites
The necessary packages (with the tested versions with Python 3.8.5) are specified in the file requirements.txt. These packages can be installed with pip:

```
pip3 install -r requirements.txt
```

---

## License and copyright

&copy; Marcos del Cueto Cordones

Licensed under the [MIT License](LICENSE.md).

