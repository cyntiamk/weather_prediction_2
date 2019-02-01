{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SQLite Database content:\n",
    "\n",
    "Database name : weather_predict.db\n",
    "\n",
    "- Tables with features created from bulk JSON files to train regression models:\n",
    "\n",
    "    1. kyoto_train_feats\n",
    "    2. manly_train_feats\n",
    "    3. nice_train_feats\n",
    "    4. kauai_train_feats\n",
    "    5. salvador_train_feats\n",
    "    \n",
    "- Tables with hourly measurements from the OpenWeatherMap API(updated every 60 minutes):\n",
    "\n",
    "    1. kyoto_recent\n",
    "    2. manly_recent\n",
    "    3. nice_recent\n",
    "    4. kauai_recent\n",
    "    5. salvador_recent\n",
    "    6. amsterdam_recent\n",
    "    7. irvine_recent\n",
    "    \n",
    "- Tables with features created from hourly measurements to get predictions from the model:\n",
    "\n",
    "    1. Kyoto_recent_features\n",
    "    2. Manly_recent_features\n",
    "    3. Nice_recent_features\n",
    "    4. Kauai_recent_features\n",
    "    5. Salvador_recent_features\n",
    "    6. Amsterdam_recent_features\n",
    "    7. Irvine_recent_features\n",
    "    \n",
    "- Tables with predictions vs actual temperatures:\n",
    "\n",
    "    1. Kyoto_prediction\n",
    "    2. Manly_prediction\n",
    "    3. Nice_prediction\n",
    "    4. Kauai_prediction\n",
    "    5. Salvador_prediction\n",
    "    6. Amsterdam_prediction\n",
    "    7. Irvine_prediction\n",
    "    \n",
    "- Tables with features for the next day prediction:\n",
    "\n",
    "    1. Kyoto_tomorrows_features\n",
    "    2. Manly_tomorrows_features\n",
    "    3. Nice_tomorrows_features\n",
    "    4. Kauai_tomorrows_features\n",
    "    5. Salvador_tomorrows_features\n",
    "    6. Amsterdam_tomorrows_features\n",
    "    7. Irvine_tomorrows_features\n",
    "\n",
    "- Tables with predictions for the next day only:\n",
    "\n",
    "    1. Kyoto_tomorrow\n",
    "    2. Manly_tomorrow\n",
    "    3. Nice_tomorrow\n",
    "    4. Kauai_tomorrow\n",
    "    5. Salvador_tomorrow\n",
    "    6. Amsterdam_tomorrow\n",
    "    7. Irvine_tomorrow\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:PythonData]",
   "language": "python",
   "name": "conda-env-PythonData-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
