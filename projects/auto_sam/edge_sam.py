import os
import cv2
import traceback
import numpy as np
import yaml
from sam import EdgeSAMONNX


class EdgeSAM:
    """Segmentation model using EdgeSAM"""
    default_output_mode = "polygon"

    def __init__(self, model_config) -> None:
        self.set_output_mode(self.default_output_mode)
        if isinstance(model_config, str):
            print(model_config)
            if not os.path.isfile(model_config):
                raise FileNotFoundError("Model", "Config file not found: {model_config}"
                    ).format(model_config=model_config)
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError("Unknown config type: {type}").format(type=type(model_config))
            
        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        
        # Load models
        self.target_length = self.config.get("target_length", 1024)
        self.model = EdgeSAMONNX(
            encoder_model_abs_path, decoder_model_abs_path, self.target_length
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

    def get_model_abs_path(self, model_config, model_path_field_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_path = model_config[model_path_field_name]

        # Model path is a local path
        model_abs_path = os.path.abspath(model_path)
        if not os.path.exists(model_abs_path):
            raise FileNotFoundError("Model", "Model path not found: {model_path}"
                    ).format(model_path=model_path)
        return model_abs_path

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, image=None):
        """
        Post process masks
        """
        # Find contours
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = []
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.append((point[0], point[1]))
                shapes.append(shape)
        return shapes

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """

        # if image is None or not self.marks:
        #     return AutoLabelingResult([], replace=False)

        shapes = []
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        print(cv_image.shape)
        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    filename,
                    image_embedding,
                )
            masks = self.model.predict_masks(image_embedding, self.marks)
            shapes = self.post_process(masks, cv_image)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result
    
    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode

   
            
if __name__ == "__main__":
    model_config = "/Users/huangquanjin/Code/X-AnyLabeling/projects/SAM/EdgeSAM/edge_sam.yaml"
    edge_sam = EdgeSAM(model_config)
    edge_sam.set_output_mode("polygon")
    # edge_sam.set_auto_labeling_marks([(100, 100), (200, 200)])
    # prompt = [{'type': 'point', 'data': [277, 356], 'label': 1}]
    prompt = [{'type': 'point', 'data': [400, 1060], 'label': 1}]
    edge_sam.set_auto_labeling_marks(prompt)
    result = edge_sam.predict_shapes(None, "demo.jpg")
    print(result.shapes)
    # 将这些polygon画到demo.jpg上，将上面prompt中2个点画到demo.jpg上，要明显
    image = cv2.imread("demo.jpg")
    for shape in result.shapes:
        cv2.polylines(image, [np.array(shape)], True, (0, 0, 255), 2)
    for point in prompt:
        cv2.circle(image, (point['data'][0], point['data'][1]), 10, (0, 255, 0), -1)
    cv2.imwrite("demo_result.jpg", image)