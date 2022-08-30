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

## 🙋🏻‍♂️ `whoami`

- 📍 Principal Data Scientist, DSAI, Moderna
- 🎓 ScD, MIT Biological Engineering.
- 🧬 Inverse protein, mRNA, and molecule design.

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
pytest
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

---

### 👆 What kind of tests exist?

----

#### 1️⃣ Unit Test

A test that checks that an individual function works correctly.

_Strive to write this type of test!_

----

#### 2️⃣ Execution Test

A test that only checks that a function executes without erroring.

_Use only in a pinch._

----

#### 3️⃣ Integration Test

A test that checks that multiple functions work correctly together.

_Used to check that a system is working properly._

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

1. Our model accepts the correct inputs and outputs.
2. Our model and datamodules work together.
3. Our model does not fail in training loop.

----

#### 🟦 Model input/output shapes

```python
from jax import random, vmap, numpy as np

def test_model_shapes():
    key = random.PRNGKey(55)
    num_samples = 7
    num_input_dims = 211
    inputs = random.normal(shape=(num_samples, num_input_dims))
    model = Model(num_input_dims=num_input_dims)
    outputs = vmap(model)(inputs)
    assert outputs.shape == (num_samples, 1)
```

----

#### 🤝 Model and DataModules work together

```python
def test_model_datamodule_compatibility():
    dm = DataModule()
    model = Model()
    x, y = next(iter(dm.train_dataloader()))
    pred = vmap(model)(x)
    assert x.shape == y.shape
```

----

#### ⭕️ Ensure no errors in training loop

```python
def test_model():
    model = Model()
    dm = DataModule()
    trainer = default_trainer(epochs=2)
    trainer.fit(model, dm)
```

Ensure that model can be trained for at least 2 epochs.

---

### 📀 Testing Data

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
    df_schema.validate(df)
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
    d1 = func1(data)
    d2 = func2(d1)
    d3 = func3(d1)
    d4 = func4(d2, d3)
    return outfunc(d4)
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

## 😎 Summary

1. ✅ Write tests for your **code**.
2. ✅ Write tests for your **data**.
3. ✅ Write tests for your **models**.

---

## Thank you! 😁
