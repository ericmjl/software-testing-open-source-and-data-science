---
theme: "black"
title: Test your (data science) work!
revealOptions:
  transition: 'fade'
---

<!-- markdownlint-disable MD035 -->
<!-- markdownlint-disable MD025 -->
<!-- markdownlint-disable MD026 -->
# Test your work!
<!-- markdownlint-enable MD026 -->

<!-- TODO: Turn this into a tinyurl -->
`https://tinyurl.com/test-sdm`

---

## 🤔 My goals today

- To share hard-won lessons borne out of failures in my past.
- To encourage you to embrace testing as part of your workflow.

---

## 🙋🏻‍♂️ `whoami`

- 📍 Principal Data Scientist, DSAI, Moderna
- 🎓 ScD, MIT Biological Engineering.
- 🧬 Inverse protein, mRNA, and molecule design.
- 🧠 Accelerate and enrich insights from data.

---

## 🕝 tl;dr

If you write automated tests for your work, then:

- ⬆️ Your work quality will go up.
- 🎓 Your work will become trustworthy.

---

<!-- markdownlint-disable MD026 -->
## 👀 also...
<!-- markdownlint-enable MD026 -->

- Tests apply to all software.
- Data science work is software work.
- Tests apply to data science.

---

## ⭕️ Outline

- Testing in Software
- Testing in Data Science

---

## 💻 Testing in Software

- 🤔 Why do testing?
- 🧪 What does a test look like?
- ✍️ How do I make the test _automated_?
- 💰 What benefits do I get?
- 👆 What kinds of tests exist?

---

### 🤔 Why do testing?

Tests help falsify the hypothesis that our code _works_.

----

Without testing, we will have untested assumptions about whether our code works.

----

### 🧪 What does a test look like?

----

#### ➡️ Given a function

```python
def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_columns = []
    for column in df.columns:
        column = (
            str(column)
            .lower()
            .replace(" ", "_")
            .strip("_")
        )
        cleaned_columns.append(column)
    df.columns = cleaned_columns
    return df
```

----

#### ➡️ We test for expected behaviour

```python
def test_clean_names():
    # Arrange
    df = pd.DataFrame(
        columns=["Apple", "banana", "Cauliflower Sunshine"]
    )

    # Act
    df_cleaned = clean_names(df)

    # Assert
    assert list(df_cleaned.columns) == \
        ["apple", "banana", "cauliflower_sunshine"]

    # Cleanup: nothing needed in this case
```

Read: [Anatomy of a Test](https://docs.pytest.org/en/7.1.x/explanation/anatomy.html)

---

### ✍️ How do I make tests automated?

----

#### 📦 Install `pytest`

Update your environment configuration:

```yaml
name: project_env  # your project environment!
channels:
- conda-forge
dependencies:
- python>=3.9
- ...
- pytest>=7.1  # add an entry here!
```

Then run:

```bash
mamba env update -f environment.yml
```

----

#### 🏃‍♂️ Run `pytest`

With `pytest` installed, use it to run your tests:

```bash
cd /path/to/my_project
conda activate my_project
pytest .
```

---

### 💰 What benefits do I get?

----

#### 🚇 Changes happen

```python
from string import punctuation
import re

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_columns = []
    for column in df.columns:
        column = (
            str(column)
            .lower()
            .replace(" ", "_")
            .strip("_")
        )
        # 👀 CHANGE HAPPENS HERE!
        column = re.sub(punctuation, "_", column)
        cleaned_columns.append(column)
    df.columns = cleaned_columns
    return df
```

----

#### ✅ Guarantee expectations

```bash
pytest
```

If the test fails, we _falsify_ our assumption
that the change does not break expected behaviour.

----

#### 💡 Update exepectations

```python
def test_clean_names():
    # Arrange
    df = pd.DataFrame(
        # 👀 change made here!
        columns=["Apple.Sauce", "banana", "Cauliflower Sunshine"]
    )

    # Act
    df_cleaned = clean_names(df)

    # Assert
    assert list(df_cleaned.columns) == \
        # 👀 change made here!
        ["apple_sauce", "banana", "cauliflower_sunshine"]

    # Cleanup: nothing needed here
```

We update the test to establish new expectations.

----

#### 💰 Benefits of Testing

1. ✅ Guarantees against breaking changes.
2. 🤔 Example-based documentation for your code.

> Testing is a contract between yourself (now) and yourself (in the future).

---

### 👆 What kind of tests exist?

----

#### 1️⃣ Unit Test

```python
def func1(data):
    ...
    return stuff

def test_func1(data):
    stuff = func1(data)
    assert stuff == ...
```

_A test that checks that an individual function works correctly. Strive to write this type of test!_

----

#### 2️⃣ Execution Test

```python
def func1(data):
    ...
    return stuff

def test_func1(data):
    func1(data)
```

_A test that only checks that a function executes without erroring. Use only in a pinch._

----

#### 3️⃣ Integration Test

```python
def func1(data):
    ...
    return stuff

def func2(data):
    ...
    return stuff

def pipeline(data):
    return func2(func1(data))

def test_pipeline(data):
    output = pipeline(data)
    assert output = ...
```

_Checks that a system is working properly. Use this sparingly if the tests are long to execute!_

---

<!-- markdownlint-disable MD026 -->
## 🧔‍♂️ Hadley says...
<!-- markdownlint-enable MD026 -->

<!-- markdownlint-disable MD033 -->
<iframe width="560" height="315" src="https://www.youtube.com/embed/cpbtcsGE0OA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<!-- markdownlint-enable MD033 -->

_You can't do data science in a GUI..._

----

### 💻 Data science needs code

```python
>>> code == software
True-ish
```

_...implying that you'll be writing some kind of software to do data science work!_

----

### 👀 Test your code

Testing your DS code will be good for you!

---

## 😎Testing in Data Science

- Machine Learning Model Code
- Data
- Pipelines

----

### 🧠 Testing Machine Learning Model Code

```python
from project.models import Model
from project.data import DataModule
from project.trainers import default_trainer


model = Model()
dm = DataModule()
trainer = default_trainer()
trainer.fit(model, dm)
```

----

#### 👆 What do we need guarantees on?

```python
model = Model()
dm = DataModule()
```

`dm` must serve up tensors
of the shape that `model` accepts.

----

#### 🤔 What can we test here?

1. ___Unit test:___ `dm` produces correctly-shaped outputs when executed.
2. ___Unit test:___ Given random inputs, `model` produces correctly-shaped outputs.
3. ___Integration test:___ Given `dm` outputs, `model` produces correctly-shaped outputs.
4. ___Execution test:___ `model` does not fail in training loop with `trainer` and `dm`.

----

#### 🟩 DataModule output shapes

```python
def test_datamodule_shapes():
    # Arrange
    batch_size = 3
    input_dims = 4
    dm = DataModule(batch_size=batch_size)

    # Act
    x, y = next(iter(dm.train_loader()))

    # Assert
    assert x.shape == (batch_size, data_dims)
    assert y.shape == (batch_size, 1)
```

----

#### 🟦 Model input/output shapes

```python
from jax import random, vmap, numpy as np

def test_model_shapes():
    # Arrange
    key = random.PRNGKey(55)
    batch_size = 3
    input_dims = 4
    inputs = random.normal(shape=(num_samples, input_dims))
    model = Model(input_dims=input_dims)

    # Act
    outputs = vmap(model)(inputs)

    # Assert
    assert outputs.shape == (num_samples, 1)
```

----

#### 🤝 Model and DataModules work together

```python
def test_model_datamodule_compatibility():
    # Arrange
    dm = DataModule()
    model = Model()
    x, y = next(iter(dm.train_dataloader()))

    # Act
    pred = vmap(model)(x)

    # Assert
    assert pred.shape == y.shape
```

----

#### ⭕️ Ensure no errors in training loop

```python
def test_model():
    # Arrange
    model = Model()
    dm = DataModule()
    trainer = default_trainer(epochs=2)

    # Act
    trainer.fit(model, dm)
```

---

### 📀 Testing Data

_a.k.a. Data Validation_

----

#### 👆 What data guarantees do we need?

```python
def func(df):
    # The column we need is actually present
    assert "some_column" in df.columns
    # Correct dtype
    assert df["some_column"].dtype == int
    # No null values
    assert pd.isnull(df["some_column"]).sum() == 0
    # The rest of the logic
    ...
```

----

#### 📕 Schemas to declare expectations

```python
import pandera as pa

df_schema = pa.DataFrameSchema(
    columns={
        # Declare that `some_column` must exist,
        # that it must be integer type,
        # and that it cannot contain any nulls.
        "some_column": pa.Column(int, nullable=False)
    }
)
```

----

#### 🏃‍♂️ Runtime dataframe validation

```python
def func(df):
    df = df_schema.validate(df)
    # The rest of the logic
    ...
```

Runtime validation code is abstracted out.

Code is much more readable.

---

### 🚇 Testing Pipeline Code

----

#### 💡 Pipelines are functions

```python
def pipeline(data):
    data = df_schema.validate(data)
    d1 = func1(data)
    d2 = func2(d1)
    d3 = func3(d1)
    d4 = func4(d2, d3)
    output = outfunc(d4)
    return output_schema.validate(output)
```

----

#### 👆 Each unit function can be unit tested

```python
def test_func1(data):
    ...

def test_func2(data):
    ...

def test_func3(data):
    ...

def test_func4(data):
    ...
```

----

#### 🤝 The whole pipeline can be integration tested

```python
def test_pipeline()
    # Arrange
    data = pd.DataFrame(...)

    # Act
    output = pipeline(data)

    # Assert
    assert output = ...
```

_We assume your pipeline is quick to run._

---

### 🕓 One more thing

---

### 💰 Mock-up Realistic Fake Data

----

#### ☁️ Schema Generators

<img src="https://pandera.readthedocs.io/en/stable/_static/pandera-logo.png" style="max-width: 100px; max-height: 100px">

```python
from hypothesis import given

schema = pa.DataFrameSchema(...)

@given(schema.strategy(3))
def test_func1(data):
    ...
```

----

#### 🎲 Probabilistic Modelling

<h4><img src="https://camo.githubusercontent.com/bcfd83328eafae3e264cd9b3e51fc92bb36ba5053cab6e2ec11f6bbf044f8a28/68747470733a2f2f63646e2e7261776769742e636f6d2f70796d632d646576732f70796d632f6d61696e2f646f63732f6c6f676f732f7376672f50794d435f62616e6e65722e737667" style="max-width: 200px; max-height: 200px"></img></h4>

```python
import pymc as pm

with pm.Model() as model:
    mu = pm.Normal("mu")
    sigma = pm.Exponential("sigma")
    pm.Normal("observed", mu=mu, sigma=sigma, observed=data)

    idata = pm.sample()
    idata.extend(pm.sample_posterior_predictive(idata))

# idata.posterior_predictive now contains
# simulated data that looks like your original data!
```

---

## ☁️ Philosophy

Integrating testing into your work is one manifestation of _defensive programming_.

----

### 1️⃣ Testing raises quality

- Save headaches in the long-run.
- Improve code quality.

----

### 2️⃣ Testing is other-centric

Others can:

- Feel confident about our code.
- Understand where their assumptions may be incorrect.

_Do unto others what you would have others do unto you._

---

## 💡 Resources

<table>
<tr>
<td>
<iframe width="560" height="315" src="https://www.youtube.com/embed/7NEQApSLT1U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
<small>How Software Skillsets Will Accelerate Your Data Science Work</small>
</td>
<td>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Dx2vG6qmtPs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
<small>Principled Data Science Workflows</small>
</td>
<td>
<iframe width="560" height="315" src="https://www.youtube.com/embed/5RKuHvZERLY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
<small>Testing Data Science Code</small>
</td>
</tr>
</table>

---

## 😎 Summary

1. ✅ Write tests for your __code__.
2. ✅ Write tests for your __data__.
3. ✅ Write tests for your __models__.

---

## Thank you! 😁
