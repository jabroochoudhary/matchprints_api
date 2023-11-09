from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
from matchprint_api.settings import BASE_DIR as dr
import cv2
import fingerprint_enhancer

@csrf_exempt
def enhanceImage(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            path = os.path.join(dr,"media")
            destination = open(path+"/inimage.png", 'wb')

            for chunk in uploaded_file.chunks():
                destination.write(chunk)
            destination.close()
            # Enchnacing the prints ridges
            image = cv2.imread(path+"/inimage.png")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # larger than the width of the widest ridges
            low = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)    # locally lowest grayvalue
            high = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)  # locally highest grayvalue
            o = low
            c = high
            gray = (gray - o) / (c - 0 + 1e-6)
            enhanced_image = fingerprint_enhancer.enhance_Fingerprint(gray, resize=True)

            # counting no of ridges
            _, thresholded = cv2.threshold(enhanced_image, 25,25, cv2.THRESH_BINARY)
            num_ridges, labels = cv2.connectedComponents(thresholded, connectivity=8)
            num_ridges -= 1
            enhanced_image = cv2.bitwise_not(enhanced_image)
            if len(enhanced_image.shape) == 2:
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            output_path = path+"/outimage.png"
            cv2.imwrite(output_path, enhanced_image)
            full_url = request.get_host()
            output_data = {
                "Ridges":num_ridges,
                "prints_url":full_url+"/media/outimage.png",
                "msg":"Prints Sucessfully enhanced.",
                "error":False
            }

            return JsonResponse(data=output_data,)
        else :
            return JsonResponse(data={ "msg":"Error in image uploading","error":True})
    
    return JsonResponse(data={ "msg":"Invalid request type","error":True})

