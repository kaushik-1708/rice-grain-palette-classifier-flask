import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from prepare_data import train_generator, validation_generator

# Image shape
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 5

# Load base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False  # Freeze the convolutional base

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    verbose=1
)

# Save model
model.save("rice_classifier_mobilenetv2.h5")
print("✅ Model trained and saved as rice_classifier_mobilenetv2.h5")

if __name__ == "__main__":
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=5,
        verbose=1
    )
    model.save("rice_classifier_mobilenetv2.h5")
    print("✅ Model trained and saved as rice_classifier_mobilenetv2.h5")
