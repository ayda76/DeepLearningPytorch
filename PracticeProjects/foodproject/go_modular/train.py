import data_setup, engine, model_builder,utils
from torchvision import transformsfrom timeit import default_timer as timer
from timeit import default_timer as timer 

NUM_EPOCHS = 5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

#DIRECTORIES
train_dir="data/pizza_steak_sushi/train"
test_dir="data/pizza_steak_sushi/test"

device="cuda" if torch.cuda.is_available() else "cpu"

data_transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
train_dataloader, test_dataloader, class_names=data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir, transform=data_transform,batch_size=BATCH_SIZE)



# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Train model_0 
engine.train(model=model_0,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,loss_fn=loss_fn, epochs=NUM_EPOCHS,device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model(model=model_0,target_dir="models",model_name="05_going_modular_cell_mode_tinyvgg_model.pth")
