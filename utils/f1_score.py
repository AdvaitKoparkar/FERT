from utils.metric import Metric
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class F1_score(Metric):
	def __init__(self, **kwargs):
		self.metric_name = 'F1_score'
		self.filename = kwargs.get('filename', 'plot.png')
		self.classes = kwargs.get('classes', [0,1,2,3,4,5,6])
		self.class_names = kwargs.get('class_names', ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
		self.y_true = kwargs.get('y_true', [])
		self.y_pred = kwargs.get('y_pred', [])

	def compute_values(self):
		class_dict = classification_report(self.y_true, self.y_pred, target_names = self.class_names, output_dict = True)
		precision = [class_dict[emotion]['precision'] for emotion in self.class_names]
		recall = [class_dict[emotion]['recall'] for emotion in self.class_names]
		f1_score = [class_dict[emotion]['f1-score'] for emotion in self.class_names]

		return precision, recall, f1_score

	def metric_plot(self):
		x = self.classes
		x_less = [i-0.2 for i in x]
		x_more = [i+0.2 for i in x]
		precision, recall, f1_score = self.compute_values()

		ax = plt.subplot(111)
		ax.bar(x_less, precision,width=0.2,color='b',align='center')
		ax.bar(x, recall,width=0.2,color='g',align='center')
		ax.bar(x_more, f1_score,width=0.2,color='r',align='center')

		plt.show()
		Metric.saveFigure(self, plt, self.filename)
