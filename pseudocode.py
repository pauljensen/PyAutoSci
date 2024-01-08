
factors = FactorSet()
factors.add_continuous(name="draw_angle", min=0, max=180)
factors.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
factors.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

# these should return a dataframe
Xinit = latin_hypercube_design(factors, n=12)
Xinit = maximin_design(factors, n=12)
Xinit = random_design(factors, n=12)

y = [...]  # users input the responses

model = GaussianProcess(factors)
model.train(Xinit, y)
# training includes hyperparameter tuning; updating does not

model.plot()
model.predict(Xnew)

X = plan_exploration(model)
X = plan_exploitation(model)
X = plan_expected_improvement(model)

y = ...  # users add response
model.update(X, y)

