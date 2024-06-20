from rest_framework.views import APIView, Response
from .models import Respuesta
from .services import classifier

"""
    Request API: 
    Format: [gender, age, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
    Example: 
        Request: [0,19,4,3,5,3,4,3,3,3,4,3,5,3,4,5,3] 
        Response: Kinesthetic
"""
class ClassifierView(APIView):
    def post(self, request, *args, **kwargs):
        try: 
            if type (request.data) != list:
                raise Exception('Input must be a list')
            if request.data.__len__() != 17:
                raise Exception('Input size must be 17')
            if request.user is None:
                raise Exception('User must be logged in')

            classify_result = classifier.classify(request.data)
            Respuesta.objects.create(
                gender = request.data[0],
                age = request.data[1],
                q1 = request.data[2], q2 = request.data[3], q3 = request.data[4], q4 = request.data[5],
                q5 = request.data[6], q6 = request.data[7], q7 = request.data[8], q8 = request.data[9],
                q9 = request.data[10], q10 = request.data[11], q11 = request.data[12], q12 = request.data[13],
                q13 = request.data[14], q14 = request.data[15], q15 = request.data[16],
                learn=classify_result,
                quizz_id=23
            )
            return Response({"result": classify_result})
        except Exception as e:
            return Response({"error": str(e)})