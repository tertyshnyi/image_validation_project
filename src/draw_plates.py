import cv2

def draw_plates(image_path, plates, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "plate", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)
