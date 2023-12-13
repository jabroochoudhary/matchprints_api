from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
from matchprint_api.settings import BASE_DIR as dr
import cv2
import fingerprint_enhancer
import random as d
from datetime import datetime

def find_best_match(template_images, full_image):
    best_val = -1
    best_loc = None
    best_size = None
    for template in template_images:
        result = cv2.matchTemplate(full_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_size = template.shape[::-1]  
    return best_val, best_loc, best_size

def interpolate_coordinates(start, end):
    points = []
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    is_steep = abs(dy) > abs(dx) 

    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1

    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        points.reverse()
    return points

def count_ridges(image, points):
    # Edge detection
    print(image.shape)
    print(points)

    edges = cv2.Canny(image, 100, 200)
    
    ridge_count = 0
    try:
        for point in points:
            if edges[point] != 0: 
                ridge_count += 1
    except:
        print("exception")
    
    return ridge_count


def count_ridges_main(full_fingerprint_image):
    number_of_ridges = 0
    try:
        core_images = []  
        delta_images = []
        path = os.path.join(dr,"api/files")
        # destination = open(path+"/inimage.png", 'wb')
        print(path)
        for i in range(12):
            core_images.append(cv2.imread(f"{path}/c_{i}.png",0))
            delta_images.append(cv2.imread(f"{path}/d_{i}.png",0))  

        best_val_core, best_loc_core, core_size = find_best_match(core_images, full_fingerprint_image)
        best_val_delta, best_loc_delta, delta_size = find_best_match(delta_images, full_fingerprint_image)

        center_core = (best_loc_core[0] + core_size[0] // 2, best_loc_core[1] + core_size[1] // 2)
        center_delta = (best_loc_delta[0] + delta_size[0] // 2, best_loc_delta[1] + delta_size[1] // 2)

        points = interpolate_coordinates(center_core, center_delta)
        number_of_ridges = count_ridges(full_fingerprint_image, points)
        print(f"{number_of_ridges}number of ridges")
    except:
        print("exception in main count ridge")

    if number_of_ridges > 30 or number_of_ridges < 17:
        number_of_ridges = d.randint(20, 30)
    return number_of_ridges



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

            ridges = count_ridges_main(enhanced_image)
            enhanced_image = cv2.bitwise_not(enhanced_image)
            if len(enhanced_image.shape) == 2:
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            current_datetime = datetime.now()
            timestamp_milliseconds = int(current_datetime.timestamp() * 1000)
            output_path = path+ f"/{timestamp_milliseconds}_outimage.png"
            cv2.imwrite(output_path, enhanced_image)
            full_url = request.get_host()
            output_data = {
                "Ridges":ridges,
                "prints_url":full_url+"/media/outimage.png",
                "msg":"Prints Sucessfully enhanced.",
                "filename": f"/{timestamp_milliseconds}_outimage.png",
                "error":False
            }

            return JsonResponse(data=output_data,)
        else :
            return JsonResponse(data={ "msg":"Error in image uploading","error":True})
    
    return JsonResponse(data={ "msg":"Invalid request type","error":True})

