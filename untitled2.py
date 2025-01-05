import ezdxf


def create_dxf_with_layer(input_dxf, target_layer, output_dxf):
    try:
        doc = ezdxf.readfile(input_dxf)
        new_doc = ezdxf.new()
        new_msp = new_doc.modelspace()

        # Copy entities from modelspace
        for entity in doc.modelspace().query(f'*[layer=="{target_layer}"]'):
            new_msp.add_entity(entity.copy())

        # Copy entities from blocks
        for block in doc.blocks:
            if block.name != "*Model_Space":
                for entity in block.query(f'*[layer=="{target_layer}"]'):
                    new_msp.add_entity(entity.copy())

        new_doc.saveas(output_dxf)

    except FileNotFoundError:
        print(f"Error: Input file '{input_dxf}' not found.")
    except ezdxf.DXFError as e:
        print(f"DXF error: {e}")

# Example usage:
create_dxf_with_layer("4-Layer-7628.dxf", "Top", "output_layer.dxf")