A logistic regression classifier that predicts either "yes, it's a cat" (1) or "no, it's not a cat" (0). Inspired by Andrew Ng's Deeplearning.ai course.

##TO-DO:

Training accuracy is near 100%, but test error is at about 70%. This means that the model is clearly overfitting the training data and lacks the ability to generalize. This may be because the dataset was too small, or because data wasn't properly regularized. Some hyperparameter tweaking is needed.

#TO START:

```
sudo apt-get install python3
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
sudo chmod +x logistic_regression_model.py
./logistic_regression_model.py
```
