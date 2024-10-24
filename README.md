<h1><center><font size=10>Data Science and Business Analytics</center></font><p
<center>Project 4 - Supervised Learning - Classification: Cancellations Predition Model for INN Hotels</center></h1><p

---

**`| Supervised Learning | Data Visualization | Statistical Analysis | Python | Data Cleaning | Univariate Analysis | Multivariate Analysis | Data Preprocessing | Logistic Regression | Multicollinearity | AUC-ROC Curve | Decision Tree Pruning |`**

<p align="left"> 
  <a href="https://github.com/RayVazcari?tab=followers">
    <img alt="followers" title="Follow me on Github" src="https://custom-icon-badges.demolab.com/github/followers/RayVazcari?color=236ad3&labelColor=1155ba&style=for-the-badge&logo=person-add&label=Follow me on Github &logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/rayvazcari/">
    <img alt="Linkedin Profile" title="Linkedin Profile" src="https://custom-icon-badges.demolab.com/badge/-Linkedin%20Profile-blue?style=for-the-badge&logoColor=white&logo=linkedin"/></a>
</p>

---

### 🧰 Languages Libraries and Tools I Used on This Project
<a href="https://jupyter.org/" target="_blank"><img align="left" alt="Jupyter" title="Jupyter" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/jupyter/jupyter-original-wordmark.svg" /></a>
<a href="https://matplotlib.org/" target="_blank"><img align="left" alt="Matplotlib" title="Matplotlib" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" /></a>
<a href="https://numpy.org/" target="_blank"><img align="left" alt="Numpy" title="Numpy" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" /></a>
<a href="https://pandas.pydata.org/" target="_blank"><img align="left" alt="Pandas" title="Pandas" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" /></a>
<a href="https://plotly.com/" target="_blank"><img align="left" alt="Plotly" title="Plotly" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/plotly/plotly-original.svg" /></a>
<a href="https://www.python.org/" target="_blank"><img align="left" alt="Python" title="Python" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" /></a>
<a href="https://www.raspberrypi.org/" target="_blank"><img align="left" alt="Raspberry Pi" title="Raspberry Pi" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/raspberrypi/raspberrypi-original.svg" /></a>
<a href="https://code.visualstudio.com/" target="_blank"><img align="left" alt="VScode" title="VScode" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/vscode/vscode-original.svg" /></a>
<a href="https://seaborn.pydata.org/" target="_blank"><img align="left" alt="Seaborn" title="Seaborn" width="30px" style="padding-right:10px;" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" /></a>

<br />

---

**Project Overview**

In this project, I analyzed the booking data of INN Hotels Group to identify key factors driving booking cancellations and developed a machine learning model to predict future cancellations. The project began with a comprehensive Exploratory Data Analysis (EDA) to uncover patterns in customer behavior, focusing on attributes that strongly correlated with cancellations. Using data visualization and statistical analysis, I explored the relationships between variables such as lead time, special requests, and market segment type, which helped me identify key predictors of cancellations.

Next, I performed Data Preprocessing techniques to prepare the dataset for modeling, including handling missing values, encoding categorical variables, and scaling numerical features. This ensured a clean and structured dataset, ready for model development.

For the modeling phase, I initially built a Logistic Regression model to establish a baseline for prediction performance, using metrics such as accuracy and the AUC-ROC curve to evaluate the model's ability to distinguish between canceled and non-canceled bookings. I then applied a Decision Tree model to improve both accuracy and interpretability, using techniques like pruning to prevent overfitting and enhance the model’s ability to generalize to new data.

**Summary of Findings**

- `Lead Time`: The most significant predictor of cancellations. Longer lead times were strongly correlated with a higher likelihood of cancellations.
- `Special Requests`: Guests who made more special requests had a higher probability of canceling, possibly due to unmet expectations.
- `Market Segment Type`: Bookings made through online channels had a higher cancellation rate compared to direct bookings.
- `Price Sensitivity`: Lower-priced bookings were more likely to be canceled, suggesting that price-sensitive customers are more prone to change their plans.

**Impact**

Based on these insights, I recommended several strategies for INN Hotels to reduce cancellations and improve profitability:
- Implement stricter cancellation policies for bookings with long lead times to mitigate the risk of cancellations closer to the booking date.
- Offer non-refundable rates at lower price points to attract price-sensitive travelers while securing revenue upfront.
- Tailor cancellation policies by booking channel, particularly for online bookings where the cancellation rate is higher.
These data-driven recommendations are designed to minimize last-minute cancellations, reduce revenue loss, and improve overall operational efficiency.

**Project Outcome**

This project allowed me to refine my skills in end-to-end data analysis and machine learning, with a particular focus on deriving actionable insights that directly address business challenges. By leveraging Logistic Regression and Decision Tree models, I was able to create a solution that not only predicts cancellations but also provides strategic guidance for policy adjustments to enhance profitability and customer satisfaction.

<br />


---

<center><img src="https://th.bing.com/th/id/R.8530a59736209430f8710ee60a231e61?rik=e2upkQ65TLGy8w&riu=http%3a%2f%2fcreativemite.com%2fimg%2flogo%2fInn-Hotels-Official-color.png&ehk=dDBg7jmsBKSMMLoEBxu3XdVy9zchx38Uluu5asTExM4%3d&risl=&pid=ImgRaw&r=0"></center>

---

---
### Business Context

A significant number of hotel bookings are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.

The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.

The cancellation of bookings impact a hotel on various fronts:

1. Loss of resources (revenue) when the hotel cannot resell the room.
2. Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
3. Lowering prices last minute, so the hotel can resell a room, resulting in reducing the profit margin.
4. Human resources to make arrangements for the guests.

### Objective

The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.

### Data Description

The data contains the different attributes of customers' booking details. The detailed data dictionary is given below.

**Data Dictionary**

- `Booking_ID`: unique identifier of each booking
- `no_of_adults`: Number of adults
- `no_of_children`: Number of Children
- `no_of_weekend_nights`: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
- `no_of_week_nights`: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
- `type_of_meal_plan`: Type of meal plan booked by the customer:
  - `Not Selected` – No meal plan selected
  - `Meal Plan 1` – Breakfast
  - `Meal Plan 2` – Half board (breakfast and one other meal)
  - `Meal Plan 3` – Full board (breakfast, lunch, and dinner)
- `required_car_parking_space`: Does the customer require a car parking space? (0 - No, 1- Yes)
- `room_type_reserved`: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
- `lead_time`: Number of days between the date of booking and the arrival date
- `arrival_year`: Year of arrival date
- `arrival_month`: Month of arrival date
- `arrival_date`: Date of the month
- `market_segment_type`: Market segment designation.
- `repeated_guest`: Is the customer a repeated guest? (0 - No, 1- Yes)
- `no_of_previous_cancellations`: Number of previous bookings that were canceled by the customer prior to the current booking
- `no_of_previous_bookings_not_canceled`: Number of previous bookings not canceled by the customer prior to the current booking
- `avg_price_per_room`: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
- `no_of_special_requests`: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
- `booking_status`: Flag indicating if the booking was canceled or not.
