import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import xml.dom.minidom
from math import isqrt

class TrafficLightXMLGenerator:
    def __init__(self, num_intersections=1, lanes_per_intersection=2, left_turn_control='no', layout='h', gap=10):
        self.num_intersections = num_intersections
        self.lanes_per_intersection = lanes_per_intersection
        self.left_turn_control = left_turn_control
        self.layout = layout
        self.gap = gap

    def generate(self, output_file):
        root = ET.Element("network")
        for i in range(self.num_intersections):
            self._add_intersection(root, i)
        # Pretty-print the XML with layout information
        xml_str = ET.tostring(root, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml_as_string = dom.toprettyxml()
        layout_comment = f'<!-- Layout: {self.layout} -->\n'
        pretty_xml_as_string = layout_comment + pretty_xml_as_string
        with open(output_file, "w") as f:
            f.write(pretty_xml_as_string)

    def _add_intersection(self, root, intersection_id):
        tlLogic = ET.SubElement(root, "tlLogic", id=f"{intersection_id}", programID="my_program", offset="0", type="actuated")
        ET.SubElement(tlLogic, "param", key="max-gap", value="3.0")
        ET.SubElement(tlLogic, "param", key="detector-gap", value="2.0")
        ET.SubElement(tlLogic, "param", key="passing-time", value="2.0")
        ET.SubElement(tlLogic, "param", key="jam-threshold", value="30")

        # Add phases based on pattern
        if self.left_turn_control == 'no':
            for _ in range(self.lanes_per_intersection):
                ET.SubElement(tlLogic, "phase", duration="31", minDur="5", maxDur="45", state="GGggrrrrGGggrrrr")
        elif self.left_turn_control == 'yes':
            for _ in range(self.lanes_per_intersection):
                ET.SubElement(tlLogic, "phase", duration="31", minDur="5", maxDur="45", state="GGGgrrrrGGGgrrrr")
                ET.SubElement(tlLogic, "phase", duration="31", minDur="5", maxDur="45", state="rrrrGGGgrrrrGGGg")

    def _closest_factors(self, n):
        for i in range(isqrt(n), 0, -1):
            if n % i == 0:
                return i, n // i

    def visualize(self, xml_file):
        fig, ax = plt.subplots()
        if self.layout == 'h':
            ax.set_xlim(0, self.gap * self.num_intersections)
            ax.set_ylim(0, self.gap * self.lanes_per_intersection)
        elif self.layout == 'g':
            rows, cols = self._closest_factors(self.num_intersections)
            ax.set_xlim(0, self.gap * cols)
            ax.set_ylim(0, self.gap * rows)

        for i in range(self.num_intersections):
            if self.layout == 'h':
                intersection_center = (self.gap/2 + i * self.gap, self.gap/2 + self.lanes_per_intersection / 2)
            elif self.layout == 'g':
                intersection_center = (
                    self.gap/2 + (i % cols) * self.gap + (self.lanes_per_intersection - 1) / 2,
                    self.gap/2 + (i // cols) * self.gap + (self.lanes_per_intersection - 1) / 2
                )

            # Add intersection 
            if self.lanes_per_intersection == 1:
                ax.add_patch(plt.Rectangle((intersection_center[0], intersection_center[1]), self.lanes_per_intersection, self.lanes_per_intersection, fill=True, color='orange'))
            else:
                ax.add_patch(plt.Rectangle((intersection_center[0], intersection_center[1]), self.lanes_per_intersection - 1, self.lanes_per_intersection - 1, fill=True, color='orange'))

            # Draw lanes meeting at the intersection 
            for j in range(self.lanes_per_intersection):
                # Horizontal lanes
                ax.plot([intersection_center[0] - self.gap/2, intersection_center[0] + self.gap/2], [intersection_center[1] + j, intersection_center[1] + j], color='black', linewidth=2)
                # Vertical lanes
                ax.plot([intersection_center[0] + j, intersection_center[0] + j], [intersection_center[1] - self.gap/2, intersection_center[1] + self.gap/2], color='black', linewidth=2)

        ax.set_aspect('equal')
        plt.title('Traffic Light System Visualization')
        plt.xlabel('Distance')
        plt.ylabel('Distance')

        # Store PNGs
        png_output_dir = "generated_configs_png"
        os.makedirs(png_output_dir, exist_ok=True)
        png_file_path = os.path.join(png_output_dir, f"{os.path.splitext(os.path.basename(xml_file))[0]}.png")
        plt.savefig(png_file_path)
        print(f"Generated PNG file: {png_file_path}")

        plt.show()

if __name__ == "__main__":
    num_intersections = int(input("Enter the number of intersections: "))
    lanes_per_intersection = int(input("Enter the number of lanes per intersection: "))
    gap = int(input("Enter the average distance/gap between intersections: "))
    pattern = input("Would you like left-turn control? (yes/no): ")
    layout = input("Choose layout (h for horizontal, g for grid): ")
    
    output_dir = "generated_configs"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/inters_{num_intersections}_lanes_{lanes_per_intersection}_gap_{gap}_ltc_{pattern}_layout_{layout}.xml"
    generator = TrafficLightXMLGenerator(num_intersections=num_intersections, lanes_per_intersection=lanes_per_intersection, left_turn_control=pattern, layout=layout, gap=gap)
    generator.generate(output_file)
    generator.visualize(output_file)
    print(f"Generated XML file: {output_file}")
