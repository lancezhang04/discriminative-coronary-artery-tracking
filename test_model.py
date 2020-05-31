N = 100

if __name__ == "__main__":
    fake_labels = create_fake_labels(N=N)
    fake_images = create_fake_images(N=N)
    print("fake_labels loaded, shape: " + str([fake_labels[i].shape for i in range(3)]))
    print("fake_images loaded, shape: " + str(fake_images.shape) + "\n")

    model = create_model(initial_lr=0.1)
    print("model successfully created")
    input_ = input("view model structure? (Y/N): ")
    if input_ == "Y":
        model.summary()

    preds = model.predict(fake_images)
    print("sample prediction of %d images completed, shape: " % N + str([preds[i].shape for i in range(3)]) + "\n")

    input_ = input("test model for 20 epochs on fake examples? (Y/N): ")
    if input_ == "Y":
        print("")
        model.fit(fake_images, fake_labels, epochs=20, verbose=1)
