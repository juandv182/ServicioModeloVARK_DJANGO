from django.db import models

class Respuesta (models.Model):
    gender = models.IntegerField()
    age = models.IntegerField()
    q1 = models.IntegerField()
    q2 = models.IntegerField()
    q3 = models.IntegerField()
    q4 = models.IntegerField()
    q5 = models.IntegerField()
    q6 = models.IntegerField()
    q7 = models.IntegerField()
    q8 = models.IntegerField()
    q9 = models.IntegerField()
    q10 = models.IntegerField()
    q11 = models.IntegerField()
    q12 = models.IntegerField()
    q13 = models.IntegerField()
    q14 = models.IntegerField()
    q15 = models.IntegerField()
    learn = models.CharField(max_length=100)
    quizz_id = models.IntegerField()  
    class Meta:
        db_table = 'learning_predictions'  # Nombre de la tabla en la base de datos