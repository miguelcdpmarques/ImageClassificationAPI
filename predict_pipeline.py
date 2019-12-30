import numpy as np
from keras import models
from keras.preprocessing import image 


class PredictPipeline:

	CLASS_INDICES = {
		0: 'Apple Red 1', 1: 'Apple Red Yellow 2', 2: 'Avocado', 3: 'Banana', 4: 'Blueberry', 5: 'Cactus fruit', 6: 'Cherry Wax Black', 7: 'Clementine', 8: 'Cocos', 9: 'Grape Blue', 10: 'Grape White', 11: 'Kiwi', 12: 'Lemon', 13: 'Limes', 14: 'Mango', 15: 'Nectarine', 16: 'Orange', 17: 'Peach', 18: 'Pear', 19: 'Pineapple', 20: 'Plum', 21: 'Strawberry'
	}

	def __init__(self, json_model_path, model_weights_path, img_path):
		with open(json_model_path, "r") as json_file:
			loaded_model_json = json_file.read()
		self.model = models.model_from_json(loaded_model_json)
		self.model.load_weights(model_weights_path)
		self.img_path = img_path
		
	def image_process(self):
		img = image.load_img(self.img_path, target_size=(100, 100))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		images = np.vstack([x])
		images /= 255.
		return images

	def image_class_predict(self):
		class_index = self.model.predict_classes(self.image_process(), batch_size=1)
		return self.CLASS_INDICES[class_index[0]]


if __name__ == "__main__":
	pipeline = PredictPipeline(
		json_model_path='model.json', model_weights_path='model_weights.h5',img_path='test_banana.jpg'
	)
	prediction = pipeline.image_class_predict()
	print(prediction)



#Por fazer...

# fruits = [{
# 	"id": "0",
# 	"name": "Apple Red 1",
# 	"nutrition-per-100g": {
# 		"energy": 1560,
# 		"protein": 12.3,
# 		"fat": 9.9,
# 		"saturated-fat": 2.8,
# 		"carbohydrate": 51.7,
# 		"sugars": 19.7,
# 		"dietary-fibre": 13,
# 		"sodium": 6
# 	},
# 	"tags": ["grain", "carb"]
# },{
# 	"id": "0",
# 	"name": "Apple Red 1",
# 	"nutrition-per-100g": {
# 		"energy": 1560,
# 		"protein": 12.3,
# 		"fat": 9.9,
# 		"saturated-fat": 2.8,
# 		"carbohydrate": 51.7,
# 		"sugars": 19.7,
# 		"dietary-fibre": 13,
# 		"sodium": 6
# 	},
# 	"tags": ["grain", "carb"]
# }]


