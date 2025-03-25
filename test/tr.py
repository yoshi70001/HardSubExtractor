if containText :
            if len(frameBuffer) > 0:
                last_frame_data = frameBuffer[-1]
                last_contours = last_frame_data["contours"]
                
                # Comparar los contornos actuales con los anteriores
                if not compare_contours(contours, last_contours):                    
                    # Guardar la imagen con el tiempo de inicio y fin
                    if len(frameBuffer) > 5:
                        frameBuffer.pop(0)
                    
                    # Guardar el tiempo de inicio si es el primer frame del nuevo conjunto
                    if "start_time" not in frameBuffer[-1]:
                        frameBuffer[-1]["start_time"] = frameBuffer[-1]["time"]
                    
                    # Guardar el tiempo de fin para el frame anterior
                    frameBuffer[-1]["end_time"] = counters[i]
                    
                    # Guardar la nueva imagen en el buffer
                    frameBuffer.append({"image": frames[i], "contours": contours, "time": counters[i]})
                    
                    # Guardar la imagen anterior con el tiempo de inicio y fin
                    previous_frame_data = frameBuffer[-2]

                    # scaled_contours = scale_contours(last_frame_data["contours"], original_width, original_height)
                    # image_with_contours = draw_contours_on_image(previous_frame_data["image"], scaled_contours)
                    start_time = previous_frame_data["start_time"]
                    end_time = previous_frame_data["end_time"]
                    filename = f"./frames/{start_time}_{end_time}.jpeg"
                    cv2.imwrite(filename, previous_frame_data["image"])
                    # cv2.imwrite(filename, image_with_contours)
                    print(f'Escribiendo {filename}')
            else:
                # Guardar la primera imagen en el buffer
                if len(frameBuffer) > 5:
                    frameBuffer.pop(0)
                frameBuffer.append({"image": frames[i], "contours": contours, "time": counters[i]})