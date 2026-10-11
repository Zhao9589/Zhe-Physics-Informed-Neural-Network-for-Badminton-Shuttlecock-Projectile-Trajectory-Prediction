# Zhe-Physics-Informed-Neural-Network-for-Badminton-Shuttlecock-Projectile-Trajectory-Prediction
Full project code: `Zhe-physicsPINN.ipynb`
## Abstract
This project uses Euler simulation to create datasets of badminton flight paths. It compares the prediction results of traditional machine learning models and a Physics-Informed Neural Network (PINN). I train the models with data at angles from 30° to 45°. Then I test how well they work within this training range, and how they handle new large angles from 50° to 60°. Traditional models fail when predicting these new cases. With rules from physics equations, PINN can still draw flight paths that make physical sense for unseen situations. However, there are still some errors in its predicted flight range.

## Project Background
I’ve played badminton for 11 years. When I watch shuttlecocks fly during smashes, I notice they don’t follow the simple projectile path we learn in physics class. The strong, nonlinear air drag changes the trajectory a lot. This real-life observation got me thinking, so I made this project. I use a Physics-Informed Neural Network (PINN) to simulate how a shuttlecock flies.

A badminton shuttlecock has nonlinear air drag. This makes its movement harder to calculate than simple ideal projectile motion. Euler simulation can calculate accurate flight paths, but it takes a lot of time to test many different hitting settings.

Normal machine learning models only learn patterns from the data they are given. When we test conditions outside the training data, their predictions often break the rules of physics. To solve this problem, I built a PINN. PINN adds physics rules to the training process. So it may still give reasonable predictions even for situations it has never seen.

## Project Goal
This project aims to predict the flight paths and landing distances of badminton shuttlecocks. I use simulation data to train linear regression, random forest, a simple MLP and PINN. I test and compare their results on data inside and outside the training range. I want to find out whether PINN can make better predictions for new, unseen conditions than ordinary machine learning models.

## Project Workflow & Principles
### Step 1: Generate simulation datasets with Euler method
**What I did**: I wrote an Explicit Euler Method simulation program to calculate shuttlecock flight under different starting conditions: launch speed, angle and drag coefficient. Three CSV files are created.
**Principle**: The Euler method solves Newton’s second law step by step. It calculates acceleration caused by gravity and quadratic air drag, then updates speed and position with small time steps. This gives reliable reference flight data.

Three generated CSV files:
1. `range_dataset.csv`: 3000 simulation samples. It stores starting physical parameters (v0, angle, mass, drag_coeff), ideal projectile range and real landing range. Initial speed v0 ranges from 20 m/s to 40 m/s. Training launch angles are 30°‑45°, covering hits from amateur to professional players. Drag coefficient varies from 0.0009 to 0.0013. This dataset is used for traditional machine learning models (linear regression and random forest). I split it into training set and test data within the training range.
2. `traj_observation.csv`: Observation points for the PINN model. For each flight path, I evenly pick 25 points and record time `t`, position `x` and `y`.
3. `extrapolation_range_test.csv`: 600 test samples outside the training range. The launch angles are between 50–60°, higher than the training angles of 30–45°. This set tests whether the model can predict results under new, unknown physical conditions.

### Step 2: Exploratory Data Analysis (EDA)
**What I did**: I draw histograms, correlation heatmaps and scatter plots to check data distribution and relationships between variables.
**Principle**: EDA helps me understand the dataset. It shows the difference between training data and out-of-range test data. It also compares ideal flight range (no air drag) with real flight range (with air drag).

### Step 3: Train baseline machine learning models
**What I did**: Train linear regression, random forest and a simple MLP using `range_dataset.csv`. The input features are v0, angle, mass and drag_coeff. The target output is landing range.
**Principle**: These models only find mathematical relationships from input and output data. They do not know the physics equations of shuttlecock movement. They fit data they have seen well, but perform poorly for completely new conditions outside the training range.
I test several sizes of hidden layers for the simple MLP, and choose hidden dimension = 64 as the main comparison model.

### Step 4: Train the PINN model
**What I did**: Use the same fully connected network structure as the simple MLP. But the loss function has three parts: data loss, initial condition loss and physics loss. After training, the model file is saved as `pinn_badminton_fixed.pt`.
**Principle**:
1. Data loss: makes the model’s prediction match the sampled points on flight paths.
2. Initial condition loss: makes the prediction follow the shuttlecock’s starting position (t=0, x=0, y=1.8 m).
3. Physics loss: uses Newton’s motion equation. It makes predicted flight paths follow acceleration rules from gravity and quadratic air resistance.
PINN learns from both sampled flight data and physics differential equations at the same time.

### Step 5: Model prediction and visualization comparison
**What I did**: Load the saved PINN model. Choose test cases the model has seen and never seen. Draw graphs to compare PINN predicted flight paths with reference results from Euler simulation. Calculate the absolute error of landing range.
**Principle**: Compare model performance for familiar parameters inside the training range and unfamiliar large angles outside the range. Check whether models produce flight paths that fit physics rules.

## Main Results
For cases seen during training, PINN creates flight shapes close to Euler simulation, but landing distance still has clear errors.
For unseen large-angle cases, traditional ML models and simple MLP produce unrealistic flight paths. PINN still outputs flight curves that follow physics, though its predicted range is usually smaller than the true value. The hardest part of training PINN is balancing the three types of loss.

## Possible Improvements
1. Add more training samples, especially for fast and large-angle hits
2. Change the weight of physics loss in the total loss function
3. Add more sample points for physics equations during PINN training

## How to run this notebook
1. Install required Python packages: `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
2. Run the code cells in order:
    1. Generate 3 CSV datasets using Euler simulation
    2. Exploratory Data Analysis and plot data figures
    3. Train baseline models: Linear Regression, Random Forest, Simple MLP
    4. Train PINN model and save model weights to `pinn_badminton_fixed.pt`
    5. Load trained PINN, predict flight paths and draw comparison graphs
3. Output files: 3 CSV data files, EDA figures, saved PINN model file `pinn_badminton_fixed.pt`
