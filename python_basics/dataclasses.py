"""
Python Dataclasses

Practice:
- @dataclass
- Default values
- field()
- __post_init__()
- Frozen dataclasses
- Comparing dataclass objects
- Converting dataclasses to dictionaries
- Nested dataclasses
- Practical configuration/data-model examples
"""

from dataclasses import (
    dataclass,
    field,
    asdict,
    astuple,
)


# ---------------------------------------------------------
# 1. Basic Dataclass
# ---------------------------------------------------------

@dataclass
class User:
    name: str
    age: int
    city: str


user = User(
    name="Alice",
    age=25,
    city="Mumbai",
)

print("1. Basic Dataclass:")
print(user)


# ---------------------------------------------------------
# 2. Accessing Fields
# ---------------------------------------------------------

print("\n2. Accessing Fields:")

print("Name:", user.name)
print("Age:", user.age)
print("City:", user.city)


# ---------------------------------------------------------
# 3. Default Values
# ---------------------------------------------------------

@dataclass
class Product:
    name: str
    price: float
    currency: str = "INR"
    in_stock: bool = True


product = Product(
    name="Keyboard",
    price=2499.0,
)

print("\n3. Default Values:")
print(product)


# ---------------------------------------------------------
# 4. Default Factory
# ---------------------------------------------------------
# Use default_factory when the default value should be
# created separately for every instance.

@dataclass
class ShoppingCart:
    items: list[str] = field(
        default_factory=list
    )


cart_a = ShoppingCart()
cart_b = ShoppingCart()

cart_a.items.append("Keyboard")

print("\n4. Default Factory:")
print("Cart A:", cart_a)
print("Cart B:", cart_b)


# ---------------------------------------------------------
# 5. Field Configuration
# ---------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

    # This field will not appear in the generated repr.
    secret_key: str = field(
        default="hidden",
        repr=False,
    )


config = ModelConfig(
    model_name="TextClassifier"
)

print("\n5. Field Configuration:")
print(config)
print("Secret Key:", config.secret_key)


# ---------------------------------------------------------
# 6. __post_init__()
# ---------------------------------------------------------
# __post_init__ runs automatically after the generated
# __init__ method.

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        if self.epochs <= 0:
            raise ValueError(
                "Epochs must be positive."
            )


print("\n6. __post_init__ Validation:")

training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
)

print(training_config)


# ---------------------------------------------------------
# 7. Dataclass Methods
# ---------------------------------------------------------

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


rectangle = Rectangle(
    width=10,
    height=5,
)

print("\n7. Methods in Dataclasses:")
print("Rectangle:", rectangle)
print("Area:", rectangle.area())
print("Perimeter:", rectangle.perimeter())


# ---------------------------------------------------------
# 8. Comparing Dataclass Objects
# ---------------------------------------------------------

@dataclass
class Point:
    x: int
    y: int


point_a = Point(10, 20)
point_b = Point(10, 20)
point_c = Point(5, 15)

print("\n8. Comparing Dataclass Objects:")

print("A == B:", point_a == point_b)
print("A == C:", point_a == point_c)


# ---------------------------------------------------------
# 9. Frozen Dataclass
# ---------------------------------------------------------
# frozen=True makes instances immutable.

@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str


model_version = ModelVersion(
    name="Classifier",
    version="1.0",
)

print("\n9. Frozen Dataclass:")
print(model_version)

try:
    model_version.version = "2.0"

except Exception as error:
    print("Cannot modify frozen object:")
    print(error)


# ---------------------------------------------------------
# 10. Convert Dataclass to Dictionary
# ---------------------------------------------------------

@dataclass
class Experiment:
    name: str
    learning_rate: float
    accuracy: float


experiment = Experiment(
    name="baseline-model",
    learning_rate=0.001,
    accuracy=0.91,
)

experiment_dict = asdict(experiment)

print("\n10. Dataclass to Dictionary:")
print(experiment_dict)


# ---------------------------------------------------------
# 11. Convert Dataclass to Tuple
# ---------------------------------------------------------

experiment_tuple = astuple(experiment)

print("\n11. Dataclass to Tuple:")
print(experiment_tuple)


# ---------------------------------------------------------
# 12. Nested Dataclasses
# ---------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    path: str
    batch_size: int


@dataclass
class PipelineConfig:
    model_name: str
    dataset: DatasetConfig
    learning_rate: float


dataset_config = DatasetConfig(
    name="customer-data",
    path="data/customers.csv",
    batch_size=64,
)

pipeline_config = PipelineConfig(
    model_name="CustomerClassifier",
    dataset=dataset_config,
    learning_rate=0.001,
)

print("\n12. Nested Dataclasses:")
print(pipeline_config)

print("Dataset Name:")
print(pipeline_config.dataset.name)


# ---------------------------------------------------------
# 13. Dataclass with Computed Field
# ---------------------------------------------------------

@dataclass
class Prediction:
    probability: float
    threshold: float = 0.5
    label: str = field(init=False)

    def __post_init__(self):
        self.label = (
            "Positive"
            if self.probability >= self.threshold
            else "Negative"
        )


prediction = Prediction(
    probability=0.82
)

print("\n13. Computed Field:")
print(prediction)
print("Predicted Label:", prediction.label)


# ---------------------------------------------------------
# 14. List of Dataclass Objects
# ---------------------------------------------------------

@dataclass
class TrainingResult:
    model_name: str
    accuracy: float
    loss: float


results = [
    TrainingResult(
        model_name="Logistic Regression",
        accuracy=0.89,
        loss=0.31,
    ),
    TrainingResult(
        model_name="Random Forest",
        accuracy=0.94,
        loss=0.21,
    ),
    TrainingResult(
        model_name="Neural Network",
        accuracy=0.96,
        loss=0.15,
    ),
]

print("\n14. List of Dataclass Objects:")

for result in results:
    print(
        f"{result.model_name}: "
        f"accuracy={result.accuracy}, "
        f"loss={result.loss}"
    )


# ---------------------------------------------------------
# 15. Dataclasses and Type Hints
# ---------------------------------------------------------
# Dataclasses work especially well with type hints.

@dataclass
class APIRequest:
    endpoint: str
    method: str = "GET"
    timeout: int = 30
    headers: dict[str, str] = field(
        default_factory=dict
    )


request = APIRequest(
    endpoint="/users",
    method="POST",
    headers={
        "Content-Type": "application/json"
    },
)

print("\n15. Dataclass with Type Hints:")
print(request)


# ---------------------------------------------------------
# 16. Practical ML Configuration
# ---------------------------------------------------------

@dataclass
class MLConfig:
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "Adam"
    use_gpu: bool = False

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than 0."
            )

        if self.epochs <= 0:
            raise ValueError(
                "Epochs must be greater than 0."
            )


ml_config = MLConfig(
    model_name="ImageClassifier",
    learning_rate=0.0005,
    batch_size=64,
    epochs=20,
    optimizer="AdamW",
    use_gpu=True,
)

print("\n16. Practical ML Configuration:")
print(ml_config)


# ---------------------------------------------------------
# 17. Why Use Dataclasses?
# ---------------------------------------------------------

print("\n17. Dataclass Takeaways:")

print("1. @dataclass reduces boilerplate code.")
print("2. Type hints make the data model clear.")
print("3. Default values simplify object creation.")
print("4. field(default_factory=...) safely creates mutable defaults.")
print("5. __post_init__ is useful for validation and derived fields.")
print("6. frozen=True creates immutable data objects.")
print("7. asdict() converts dataclasses into dictionaries.")
print("8. Dataclasses are excellent for configurations and data models.")
