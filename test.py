from utils import *

N = 100

fake_labels = create_fake_labels(N=N)
fake_images = create_fake_images(N=N)
print("fake_labels loaded, shape: " + str(fake_labels.shape))
print("fake_images loaded, shape: " + str(fake_images.shape) + "\n")

model = create_model(initial_lr=0.1)
print("model successfully created")
input_ = input("view model structure? (Y/N): ")
if input_ == "Y":
    model.summary()

preds = model.predict(fake_images)
print("sample prediction of %d images completed, shape: " % N + str(preds.shape) + "\n")

input_ = input("test model for 10 epochs on fake examples? (Y/N): ")
if input_ == "Y":
    print("")
    model.fit(fake_images, fake_labels, epochs=10, verbose=1)