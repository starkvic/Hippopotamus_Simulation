from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def simulation_status(request):
    # This is a dummy response; replace it with your simulation logic.
    return Response({"status": "running", "progress": 75})
