"""Hardware and cabinet geometry for the fixed-pose v7 RGB-D scenes."""
import math
from itertools import product
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector


REFERENCES = {
    'metal_clamp': 'https://www.norma-connects.com/en/products/norma-rsgu-rubber-lined-pipe-clips-DIN-3016',
    'wire_harness_clamp': 'https://www.mcfarlaneaviation.com/products/categories/all-products/hardware-standard/clamps/adel-clamps/',
    'rail_installation': 'https://assets.phoenixcontact.com/file/fcda5126-466b-41d4-827d-91d1a2e9bfad/media/original',
}
PANEL_NAME = 'cream equipment mounting panel'  # Retain the mounting datum used by the support audit.
CLAMP_TAG = 'round_cushioned_cable_clamp'
CABINET_TAG = 'industrial_cabinet_component'


def normalize(vector):
    return vector / np.linalg.norm(vector)


def bezier(p0, p1, p2, p3, count=16):
    t = np.linspace(0, 1, count)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3


def bounds(obj):
    points = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    return points.min(axis=0), points.max(axis=0)


class Cabinet:
    def __init__(self, renderer):
        self.renderer = renderer
        self.surface = renderer.R
        self.base = renderer.B
        self.parts = self.surface.PARTS
        self.next_part_id = max(map(int, self.parts)) + 1
        self.reserved = []
        self.clamps = []
        self.placements = []
        self.material = {}
        self.set_materials()

    def set_materials(self):
        materials = {
            'backplate': ((.155, .181, .171), .48, .53),
            'steel': ((.30, .34, .35), .40, .83),
            'clamp_metal': ((.46, .411, .323), .32, .86),
            'screw': ((.52, .55, .56), .28, .90),
            'rubber': ((.038, .036, .032), .73, 0),
            'case': ((.071, .083, .082), .49, 0),
            'case_dark': ((.037, .043, .047), .56, 0),
            'duct': ((.122, .138, .146), .49, 0),
            'module': ((.251, .268, .265), .5, 0),
            'terminal': ((.327, .341, .306), .51, 0),
            'terminal_blue': ((.076, .186, .254), .50, 0),
            'terminal_ground': ((.173, .246, .086), .49, 0),
            'release': ((.39, .164, .046), .44, 0),
            'dark': ((.019, .022, .021), .66, 0),
            'sticker': ((.66, .65, .57), .61, 0),
            'ink': ((.021, .026, .024), .59, 0),
            'green': ((.067, .26, .11), .35, 0),
        }
        grain = {'backplate': (610, .22, 34e-6, 14),
                 'steel': (4100, .12, 7e-6, 8),
                 'clamp_metal': (4900, .16, 7e-6, 9),
                 'rubber': (5300, .12, 6e-6, 1),
                 'duct': (3200, .12, 11e-6, 1)}
        for key, (color, roughness, metal) in materials.items():
            self.material[key] = self.base.mat('v7_' + key, color, roughness, metal)
            if key not in grain:
                continue
            scale, variation, height, stretch = grain[key]
            tree = self.material[key].node_tree
            bsdf = tree.nodes.get('Principled BSDF')
            tex = tree.nodes.new('ShaderNodeTexCoord')
            mapping = tree.nodes.new('ShaderNodeVectorMath')
            mapping.operation = 'MULTIPLY'
            mapping.inputs[1].default_value = (1, stretch, 1)
            noise = tree.nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = scale
            noise.inputs['Detail'].default_value = 2
            tone = tree.nodes.new('ShaderNodeValToRGB')
            tone.color_ramp.elements[0].color = (*[c*(1-variation) for c in color], 1)
            tone.color_ramp.elements[1].color = (*[c*(1+variation) for c in color], 1)
            bump = tree.nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = .20
            bump.inputs['Distance'].default_value = height
            tree.links.new(tex.outputs['Object'], mapping.inputs[0])
            tree.links.new(mapping.outputs[0], noise.inputs['Vector'])
            tree.links.new(noise.outputs['Fac'], tone.inputs[0])
            tree.links.new(tone.outputs['Color'], bsdf.inputs['Base Color'])
            tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
            tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    def register(self, obj, kind='panel_device', owner=0, tags=()):
        part_id = self.next_part_id
        self.next_part_id += 1
        obj.pass_index = part_id
        self.parts[str(part_id)] = {
            'object_name': obj.name, 'part_id': part_id, 'semantic_name': kind,
            'semantic_id': self.surface.SEM[kind], 'attached_assembly_id': owner,
            'assembly_id': 0, 'branch_id': 0, 'wire_id': 0, 'junction_id': 0,
            'target': False, 'feature_tags': list(tags),
        }
        return obj

    def box(self, name, center, size, material, bevel=.0006, **labels):
        obj = self.base.box(name, center, size, self.material[material], bevel)
        return self.register(obj, **labels)

    def cylinder(self, name, start, end, radius, material, sides=24, **labels):
        obj = self.base.cyl(name, start, end, radius, self.material[material], sides)
        return self.register(obj, **labels)

    def mesh(self, name, vertices, faces, material, bevel=0, smooth=False, **labels):
        data = bpy.data.meshes.new(name)
        data.from_pydata(np.asarray(vertices).tolist(), [], faces)
        data.update()
        bm = bmesh.new()
        bm.from_mesh(data)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(data)
        bm.free()
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        data.materials.append(self.material[material])
        if smooth:
            for face in data.polygons:
                face.use_smooth = True
        if bevel:
            modifier = obj.modifiers.new('Rounded manufactured edges', 'BEVEL')
            modifier.width = bevel
            modifier.segments = 3
            obj.modifiers.new('Weighted surface normals', 'WEIGHTED_NORMAL')
        return self.register(obj, **labels)

    def text(self, body, x, y, z, size, material='ink'):
        # Use Blender's built-in font so that the blend has no external dependencies.
        obj = self.base.text_obj(body, (x, y, z), size, 'v7_' + material)
        return self.register(obj, tags=(CABINET_TAG, 'equipment_marking'))

    def remove_replaced_parts(self):
        removed = []
        for part_id, meta in list(self.parts.items()):
            name = meta['object_name']
            if name.startswith(('metal bridge clamp', 'panel fixed clamp bracket',
                                'black panel fastener recess', 'aluminum stiffener')):
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
                removed.append(part_id)
                del self.parts[part_id]
        panel = bpy.data.objects[PANEL_NAME]
        panel.data.materials.clear()
        panel.data.materials.append(self.material['backplate'])
        # The backplate stays at the original mounting datum z=0.004 m.
        bpy.data.objects['real background wall'].data.materials[0] = self.material['case_dark']
        self.renderer.SUPPORT_PATHS = [
            entry for entry in self.renderer.SUPPORT_PATHS
            if not entry['name'].startswith('panel fixed clamp bracket')
        ]
        return len(removed)

    @staticmethod
    def sweep(polyline, profile, transform):
        tangent = np.gradient(polyline, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        local = polyline[:, None, :] + normal[:, None, :] * profile[None, :, 0, None]
        axial = np.broadcast_to(profile[:, 1], local.shape[:2])
        points = transform(np.dstack((local, axial)).reshape(-1, 3))
        n = len(profile)
        faces = [tuple(reversed(range(n)))]
        for row in range(len(polyline) - 1):
            a = row * n
            for j in range(n):
                k = (j + 1) % n
                faces.append((a+j, a+k, a+n+k, a+n+j))
        faces.append(tuple(range(len(points) - n, len(points))))
        return points, faces

    def ear(self, name, attach, overlap_level, hole_x, width, thickness, transform, **tags):
        # A real open bore is formed by joining concentric loops around the bolt.
        half_x = hole_x - attach + .00018
        half_w = width / 2
        corners = [math.atan2(b, a) % (2*math.pi) for a,b in product((-half_x, half_x), (-half_w, half_w))]
        angles = np.unique(np.r_[np.linspace(0, 2*math.pi, 56, endpoint=False), corners])
        cos, sin = np.cos(angles), np.sin(angles)
        outer_r = np.minimum(half_x/np.maximum(abs(cos), 1e-8), half_w/np.maximum(abs(sin), 1e-8))
        vertices = []
        for level in (overlap_level - thickness/2, overlap_level + thickness/2):
            for radial in (outer_r, np.full(len(angles), .00165)):
                vertices.extend(np.column_stack((hole_x + radial*cos, np.full(len(angles), level), radial*sin)))
        n = len(angles)
        faces = []
        for j in range(n):
            k = (j+1) % n
            faces.extend([(j,k,n+k,n+j), (2*n+j,3*n+j,3*n+k,2*n+k),
                          (j,2*n+j,2*n+k,k), (n+j,n+k,3*n+k,3*n+j)])
        return self.mesh(name, transform(vertices), faces, 'clamp_metal', bevel=.00009, **tags)

    def clamp(self, owner, axis, boards):
        center = axis[27]
        tangent = normalize(axis[28] - axis[26])
        forward = normalize(np.array([0., 0., 1.]) - tangent*tangent[2])
        side_axis = np.cross(forward, tangent)
        pad_width = .0100
        thickness = .00076
        width = .0089
        # The bore is fitted from evaluated jacket and seam surfaces in the clamp slab.
        # A tiny allowance models contact with the rubber; no wire or jacket mesh is moved.
        jacket = []
        deps = bpy.context.evaluated_depsgraph_get()
        for meta in self.parts.values():
            if meta['assembly_id'] != owner or meta.get('physical_branch_role') != 'trunk':
                continue
            obj = bpy.data.objects[meta['object_name']].evaluated_get(deps)
            mesh = obj.to_mesh()
            points = np.array([obj.matrix_world @ vertex.co for vertex in mesh.vertices]) - center
            obj.to_mesh_clear()
            axial = points @ tangent
            nearby = abs(axial) <= pad_width/2 + .0036
            if nearby.any():
                jacket.extend(np.linalg.norm(points[nearby] - axial[nearby, None]*tangent, axis=1))
        assert jacket, 'No evaluated jacket near fixed node'
        radius = max(jacket) - .000025
        neutral = radius + .00070 + thickness/2
        hi_tab, lo_tab = -.00042, -.00118
        flat_start = neutral + .0031
        hole_x = flat_start + .00275

        support_radius = .0030
        for sign in (1, -1):
            side = sign * side_axis
            transform = lambda local: np.asarray(local) @ np.stack((side, forward, tangent)) + center
            pad_under = transform([hole_x, lo_tab-.00024, 0.])
            foot = np.r_[pad_under[:2], .0079]
            proposed = np.array([foot, pad_under])
            if not self.renderer.clear_segment(foot, pad_under, support_radius, boards, margin=.00035):
                continue
            if any(np.all(np.abs(foot[:2] - region[:2]) < .0125) for region in self.reserved):
                continue
            break
        else:
            raise RuntimeError(f'No unobstructed panel post for clamp {owner}')

        angles = np.linspace(math.radians(24), math.radians(329), 106)
        arc = neutral * np.column_stack((np.cos(angles), np.sin(angles)))
        head_angle, tail_angle = angles[0], angles[-1]
        upper = np.array([flat_start, hi_tab])
        lower = np.array([flat_start, lo_tab])
        front_leg = bezier(arc[0], arc[0]+.0019*np.array([np.sin(head_angle), -np.cos(head_angle)]),
                           upper-np.array([.0022, 0]), upper)
        rear_leg = bezier(arc[-1], arc[-1]+.0020*np.array([-np.sin(tail_angle), np.cos(tail_angle)]),
                          lower-np.array([.0022, 0]), lower)
        path = np.concatenate((front_leg[:0:-1], arc, rear_leg[1:]))
        prefix = f'H{owner} round clamp'
        ring_tags = {'kind': 'clip_tie', 'owner': owner, 'tags': (CLAMP_TAG, 'fixed_clip', 'clamp_band')}
        w, thick = width/2, thickness/2
        metal_profile = np.array([[-thick,-w],[thick,-w],[thick,w],[-thick,w]])
        vertices, faces = self.sweep(path, metal_profile, transform)
        self.mesh(prefix + ' formed metal band', vertices, faces, 'clamp_metal', bevel=.000075, smooth=True, **ring_tags)
        # On a counterclockwise loop the left-hand sweep normal points inward.
        pad_profile = np.array([[thick+.00070,-pad_width/2],
                                [thick+.00070,pad_width/2],
                                [-thick-.00011,pad_width/2],
                                [-thick-.00011,w-.00026],
                                [thick+.00010,w-.00026],
                                [thick+.00010,-w+.00026],
                                [-thick-.00011,-w+.00026],
                                [-thick-.00011,-pad_width/2]])
        vertices, faces = self.sweep(arc, pad_profile, transform)
        rubber_tags = {**ring_tags, 'tags': (CLAMP_TAG, 'fixed_clip', 'clamp_liner')}
        self.mesh(prefix + ' folded rubber liner', vertices, faces, 'rubber', bevel=.000060, smooth=True, **rubber_tags)
        ear_tags = {**ring_tags, 'tags': (CLAMP_TAG, 'fixed_clip', 'clamp_ear')}
        self.ear(prefix+' upper bored ear', flat_start, hi_tab, hole_x, width, thickness, transform, **ear_tags)
        self.ear(prefix+' lower bored ear', flat_start, lo_tab, hole_x, width, thickness, transform, **ear_tags)
        fastener_tags = {**ring_tags, 'tags': (CLAMP_TAG, 'fixed_clip', 'clamp_fastener')}

        def at(level):
            return transform([hole_x, level, 0.])

        self.cylinder(prefix+' bolt shank', at(lo_tab-.001), at(.0015), .00149, 'screw', **fastener_tags)
        self.cylinder(prefix+' washer', at(hi_tab+thick-.00004), at(.00048), .00272, 'screw', **fastener_tags)
        self.cylinder(prefix+' hex bolt head', at(.00040), at(.00221), .00268, 'screw', sides=6, **fastener_tags)
        support_tags = {'kind': 'clip_tie', 'owner': owner, 'tags': ('physical_support','all_target_clear_route')}
        # The post reaches through the underside of the lower ear by a small assembly overlap.
        self.cylinder(prefix+' hex standoff', foot, pad_under, support_radius, 'steel', sides=6, **support_tags)
        foot_tags = {**support_tags, 'tags': ('panel_attachment',)}
        self.box(prefix+' panel foot', (*foot[:2], .006), (.014, .012, .004), 'steel', **foot_tags)
        for off in (-.00465, .00465):
            x, y = foot[:2] + np.array([off, 0.])
            self.cylinder(prefix+' foot fixing', (x,y,.0078), (x,y,.009), .00155, 'screw', **support_tags)
        self.reserved.append(foot)
        self.renderer.SUPPORT_PATHS.append({
            'name': prefix+' hex standoff', 'owner_assembly_id': owner,
            'axis_polyline_world_m': proposed.tolist(), 'radius_m': support_radius,
            'target_clearance_certified_m': .00035, 'blue_base_on_panel': False,
        })
        self.clamps.append({
            'assembly_id': owner, 'center_world_m': center.tolist(), 'axis_world': tangent.tolist(),
            'material_node': 27, 'inner_radius_m': float(radius), 'band_width_m': width,
            'liner_width_m': pad_width, 'jacket_envelope_radius_m': float(max(jacket)),
            'liner_nominal_contact_allowance_m': .000025, 'post_base_world_m': foot.tolist(),
        })

    def component_box(self, name, center, size, material, bevel=.0005, tag='cabinet_sheet_metal'):
        return self.box(name, center, size, material, bevel, tags=(CABINET_TAG, tag))

    def component_screw(self, name, x, y, z, radius=.0017, tag='panel_fastener'):
        self.cylinder(name+' washer', (x,y,z-.00014), (x,y,z+.00035), radius*1.25, 'steel', tags=(CABINET_TAG, tag))
        self.cylinder(name+' head', (x,y,z+.00028), (x,y,z+.0014), radius, 'screw', sides=6, tags=(CABINET_TAG, tag))

    def reserve(self, name, centers, size, max_z):
        # Place each built-up device on the existing plate. Entire volumes are
        # tested against all three installed harnesses and their retained fittings.
        w, h = size
        for x,y in centers:
            low = np.array([x-w/2, y-h/2, .00395])
            high = np.array([x+w/2, y+h/2, max_z])
            if any(np.all(high+.0013 >= a) and np.all(b+.0013 >= low) for a,b in self.occupied):
                continue
            self.occupied.append((low, high))
            self.placements.append({'component': name, 'envelope_min_m': low.tolist(), 'envelope_max_m': high.tolist()})
            return x,y
        raise RuntimeError(f'No clear mounting location for {name}')

    @staticmethod
    def prism(profile, sides):
        # Both cap polygons share an ordered cross-section. "sides" is a
        # callable mapping that cross-section to the two ends of the prism.
        a,b = sides(np.asarray(profile))
        vertices = np.concatenate((a,b)).tolist()
        n = len(a)
        faces = [tuple(range(n)),tuple(reversed(range(n,2*n)))]
        for i in range(n):
            j = (i+1) % n
            faces.append((i,j,n+j,n+i))
        return vertices,faces

    def din_rail(self, x, y, length, name):
        # Pierced web with symmetric returns and flanges. The sections of
        # this one rolled part have a common renderer object ID at their joins.
        vertices, faces = [], []
        def section(lo,hi,yz):
            def ends(profile):
                return tuple(np.column_stack((np.full(len(profile),x+end),
                             y+profile[:,0],profile[:,1])) for end in (lo,hi))
            points,polygons = self.prism(yz,ends)
            offset = len(vertices)
            vertices.extend(points)
            faces.extend(tuple(offset+i for i in poly) for poly in polygons)
        for sign in (-1,1):
            cross_section = [(.00205,.003975),(.01195,.003975),
                             (.01195,.00955),(.0176,.00955),
                             (.0176,.011025),(.011,.011025),
                             (.011,.005625),(.00205,.005625)]
            section(-length/2,length/2,[(sign*p,z) for p,z in cross_section])
        slots = np.arange(-length/2+.010, length/2-.008, .017)
        edges = [-length/2,*(s+d for s in slots for d in (-.0031,.0031)),length/2]
        center = [(-.002065,.003975),(.002065,.003975),(.002065,.005625),(-.002065,.005625)]
        for lo,hi in zip(edges[::2],edges[1::2]):
            section(lo,hi,center)
        self.mesh(name,vertices,faces,'steel',bevel=.00014,tags=(CABINET_TAG,'din_rail'))
        for offset in (-length/2+.006, length/2-.006):
            self.component_screw(name+' fixing', x+offset, y, .0056, .0016, 'din_rail')

    def duct(self, x, y, length=.175):
        width = .022
        box = lambda n,p,d,m='duct',b=.00038: self.component_box(n,p,d,m,b,'slotted_wiring_duct')
        box('wiring duct rear channel', (x,y,.0059), (width,length,.0038))
        box('wiring duct dark interior', (x,y,.008), (.014,length-.006,.001), 'case_dark', .00016)
        for off in (-width/2+.00125,width/2-.00125):
            # The duct base and teeth are machined into one side-wall profile.
            # This removes the coincident faces between the old separate boxes.
            lo,hi = y-length/2,y+length/2
            tooth_centers = np.arange(lo+.004,hi-.0032,.0104)
            top = [(lo,.0141)]
            for mid in tooth_centers:
                a,b = mid-.00315,mid+.00315
                top.extend(((a,.0141),(a,.0252),(b,.0252),(b,.0141)))
            top.append((hi,.0141))
            yz = [(lo,.0057),(hi,.0057),*top[::-1]]
            def ends(profile):
                return tuple(np.column_stack((np.full(len(profile),x+off+dx),profile))
                             for dx in (-.00125,.00125))
            vertices,faces = self.prism(yz,ends)
            self.mesh('molded slotted duct wall',vertices,faces,'duct',bevel=.00018,
                      tags=(CABINET_TAG,'slotted_wiring_duct'))
            for yy in tooth_centers:
                box('wiring duct finger latch', (x+off*.94,yy,.0245), (.0035,.0063,.0020))
        for off,run in ((-.35,.28),(.30,.37)):
            cy=y+off*length
            box('wiring duct snap-on lid', (x,cy,.0261), (width+.0015,length*run,.0024))
            for sx in (-1,1):
                box('wiring duct lid edge', (x+sx*.0114,cy,.0248), (.0012,length*run,.0039))
        box('wireway identity strip', (x,y+length*.30,.0275), (.013,.021,.00055), 'sticker')
        self.text('WD1', x-.0055, y+length*.30-.0018, .02783, .0045)

    def terminals(self, x, y, count=8):
        rail_length = .083
        self.din_rail(x,y,rail_length,'X12 DIN rail')
        pitch = .0072
        for i in range(count):
            xx = x + (i-(count-1)/2)*pitch
            color = 'terminal' if i < count-2 else ('terminal_blue' if i == count-2 else 'terminal_ground')
            box = lambda n,p,d,m,b=.0004: self.component_box(n,p,d,m,b,'terminal_block')
            box('rail terminal block body', (xx,y,.0157), (pitch-.00035,.032,.0110),color)
            box('terminal marker deck', (xx,y,.0249), (pitch-.0004,.011,.009),color)
            for sign, slot_y in ((-1,y-.0094),(1,y+.0094)):
                box('terminal shrouded recess floor', (xx,slot_y,.0214), (.0049,.0065,.0010),'dark',.0002)
                for side in (-1,1):
                    box('terminal recess wall', (xx+side*.00243,slot_y,.0241), (.00113,.0083,.0066),color)
                box('terminal push release', (xx, y+sign*.0141,.0231), (.0034,.0030,.0042),'release')
                self.cylinder('terminal captive screw', (xx,slot_y,.0218), (xx,slot_y,.023), .00137, 'screw',
                              sides=12,tags=(CABINET_TAG,'terminal_block'))
            box('terminal number strip', (xx,y,.0297), (.00545,.0058,.0007),'sticker',.00016)
            self.text(str(i+1),xx-.0011,y-.0013,.03008,.0035)
        for offset in (-count*pitch/2-.0028,count*pitch/2+.0028):
            self.component_box('rail end stop',(x+offset,y,.0197),(.0052,.031,.019),'case',tag='terminal_block')
            self.component_screw('DIN end stop fixing',x+offset,y,.029,.00135,'terminal_block')

    def controller(self, x, y, name, width=.054, height=.053, supply=False):
        tag = 'power_supply' if supply else 'io_module'
        box = lambda n,p,d,m,b=.0007: self.component_box(name+' '+n,p,d,m,b,tag)
        self.din_rail(x, y, width+.012, name+' DIN rail')
        box('rear DIN latch body',(x,y,.0145),(width-.003,height-.002,.0095),'case_dark')
        box('molded housing',(x,y,.0190),(width,height,.0140),'case')
        top = .0327 if supply else .0330
        box('inset faceplate',(x,y-.002,.0275),(width-.0011,height-.0065,.0043),'module')
        if supply:
            # Slotted grille sits over a recessed dark compartment. Its ribs are distinct solid surfaces.
            loc = y+.008
            box('vent opening floor',(x,loc,.0298),(width-.009,.016,.0012),'dark',.0002)
            for i in range(9):
                xx = x+(i-4)*(width-.014)/9
                box('vent rib',(xx,loc,top-.00032),( .00135,.0168,.0047),'case_dark',.00025)
            for sign in (-1,1):
                box('vent trim',(x,loc+sign*.0084,top), (width-.008,.0016,.0047),'case_dark',.0002)
            box('rating label',(x,y-.012,.0301),(width-.010,.009,.0009),'sticker',.00025)
            self.text('24 V DC',x-width/2+.009,y-.0133,.03058,.0052)
        else:
            box('face insert',(x,y+.006,.0311),(width-.006,.022,.0038),'case_dark')
            box('module ID',(x-.004,y-.009,.0300),(width-.020,.0065,.0009),'sticker',.00025)
            self.text('DI  08',x-width/2+.009,y-.0106,.03048,.0045)
            for i in range(5):
                xx=x + (i-2)*.006
                self.cylinder('I/O LED lens',(xx,y+.011,.0329),(xx,y+.011,.0341),.00112,
                              'green' if i in (0,2) else 'dark',sides=16,tags=(CABINET_TAG,tag))
            self.text('I/O',x-width/2+.005,y+.003,.03308,.0050,'sticker')
        for i in range(5):
            xx=x+(i-2)*(width-.013)/5
            yy=y-height/2+.005
            box('terminal recess',(xx,yy,.02985),(.0044,.0048,.0014),'dark',.0003)
            self.cylinder('module terminal screw',(xx,yy,.03045),(xx,yy,.0316),.00142,'screw',
                          sides=12,tags=(CABINET_TAG,tag))
        for sign in (-1,1):
            box('housing side seam',(x+sign*(width/2-.0017),y,.0289),(.0017,height-.010,.0014),'case_dark',.00027)

    def cover_frame(self):
        for sign in (-1,1):
            x=sign*.220
            self.component_box('bent cabinet sidewall',(x,0,.071),(.009,.432,.145),'case_dark',.0013)
            self.component_box('folded cabinet lip',(x-sign*.013,0,.012),(.020,.424,.015),'steel')
            for y in (-.17,-.093,-.018,.057,.132,.195):
                self.component_screw('cabinet frame fixing',x-sign*.014,y,.0195,.0023)
        for sign in (-1,1):
            self.component_box('cabinet folded horizontal flange',(0,sign*.202,.013),(.416,.014,.018),'steel')

    def build(self):
        removed_parts = self.remove_replaced_parts()
        bpy.context.view_layer.update()
        target_meshes = self.surface.surface_bvh()[0]
        for owner in (1,2,3):
            self.clamp(owner,self.surface.AXES[f'assembly_{owner}_trunk'],target_meshes)
        bpy.context.view_layer.update()
        self.occupied = [bounds(bpy.data.objects[part['object_name']]) for part in self.parts.values()
                         if part['object_name'] not in {PANEL_NAME,'real background wall'}]
        x,y = self.reserve('slotted wiring duct',
                           product((-.105,-.117,-.093,-.129),(.049,.060,.034,.022,.071)),
                           (.0245,.178),.028)
        self.duct(x,y)
        x,y = self.reserve('X12 terminal bank',
                           product((-.035,-.044,-.017,-.057,.004),(.083,.091,.096,.070,.105)),
                           (.084,.036),.032)
        self.terminals(x,y)
        x,y = self.reserve('24 V DIN power module',
                           product((.076,.067,.086,.096,.054),(-.075,-.083,-.094,-.063,-.104)),
                           (.069,.054),.036)
        self.controller(x,y,'power supply',supply=True)
        x,y = self.reserve('I/O module',
                           product((.076,.090,.054,.061,.104),(.043,.059,.033,.018,.067)),
                           (.051,.042),.036)
        self.controller(x,y,'I/O module',width=.038,height=.040)
        self.cover_frame()
        bpy.context.view_layer.update()
        self.surface.FEATURES.extend({'assembly_id': item['assembly_id'], 'feature': CLAMP_TAG,
                                      'point_world_m': item['center_world_m']} for item in self.clamps)
        bpy.context.scene['cabinet_revision'] = 'cabinet_round_clamp_v7'
        bpy.context.scene['reference_policy'] = 'Procedural model based on the supplied clamp crop and manufacturer references.'
        return {'removed_legacy_parts':removed_parts, 'clamps':self.clamps,
                'cabinet_component_envelopes':self.placements,'references':REFERENCES}


def build_cabinet(renderer):
    return Cabinet(renderer).build()
