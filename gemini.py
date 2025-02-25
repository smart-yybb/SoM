import os
import google.generativeai as genai
from PIL import Image
import json
import xml.etree.ElementTree as ET

# 配置 Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 设置模型
model = genai.GenerativeModel('gemini-pro-vision')

def analyze_parking_image(image_path):
    """
    使用 Gemini 分析停车场图片
    """
    try:
        # 加载图片
        image = Image.open(image_path)
        
        # 准备 prompt
        prompt = """Given a baseline image of a parking lot with 10 parking spaces, analyze the current image and provide the status of each parking space. The status should include whether the space is occupied or not, and if occupied, the type of truck. Note any shadows that may affect the analysis.

        Please provide your response in XML format following this structure:
        <ParkingLot>
            <ParkingSpace id="1">
                <Status>occupied_or_free</Status>
                <TruckType>type_of_truck</TruckType>
                <TruckColor>color_of_truck</TruckColor>
                <Shadow>yes_or_no</Shadow>
            </ParkingSpace>
            <!-- Repeat for each parking space -->
        </ParkingLot>

        Examples:
        1. For an occupied space:
        <ParkingSpace id="1">
            <Status>occupied</Status>
            <TruckType>pickup</TruckType>
            <TruckColor>red</TruckColor>
            <Shadow>no</Shadow>
        </ParkingSpace>

        2. For a free space:
        <ParkingSpace id="2">
            <Status>free</Status>
            <TruckType></TruckType>
            <TruckColor></TruckColor>
            <Shadow>yes</Shadow>
        </ParkingSpace>

        Important: If a space is free, the TruckType and TruckColor elements should be empty.
        Please analyze all visible parking spaces in the image."""

        # 调用 Gemini API
        response = model.generate_content([prompt, image])
        
        # 从响应中提取 XML
        try:
            # 查找响应中的 XML 部分
            xml_str = response.text
            if '```xml' in xml_str:
                xml_str = xml_str.split('```xml')[1].split('```')[0]
            elif '```' in xml_str:
                xml_str = xml_str.split('```')[1].split('```')[0]
            
            # 解析XML
            root = ET.fromstring(xml_str.strip())
            results = {}
            
            # 将XML转换为字典格式
            for space in root.findall('ParkingSpace'):
                space_id = space.get('id')
                status = space.find('Status').text
                truck_type = space.find('TruckType').text or ""
                truck_color = space.find('TruckColor').text or ""
                shadow = space.find('Shadow').text
                
                results[f"space_{space_id}"] = {
                    'status': status,
                    'truck_type': truck_type,
                    'truck_color': truck_color,
                    'shadow': shadow
                }
            return results
            
        except Exception as e:
            print(f"Error parsing XML from response: {e}")
            print(f"Raw response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None

def calculate_occupancy_rate(results):
    """
    计算停车占用率统计
    """
    if not results:
        return 0.0
        
    total_spaces = len(results)
    occupied_spaces = sum(1 for space in results.values() if space['status'].lower() == 'occupied')
    
    occupancy_rate = (occupied_spaces / total_spaces) * 100 if total_spaces > 0 else 0
    return occupancy_rate
