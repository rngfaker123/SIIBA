#!/usr/bin/env python3
"""Streaming full-clean-TEST frozen masks, compressed 1024 fractional coverage.

No >=1000 eligibility filter is applied to TEST. Empty domains remain in the
catalog and receive zero support; downstream metrics must report null and their
own valid N. Original detector/geometry/company files are never edited. Native
stage artifacts are private temporaries; their audit ledger is archived before
the exact owned directory is removed. A sealed per-case receipt is published
only after the compressed coverage and evidence are complete.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
CONTROLLER_SHA = "0c47a4f306749ac7913d5e8e9132e8b77df481651da5ffc15e728eaebd289c1a"
BUILDER_SHA = "aa05bbca42235676637fea1cb5fa53832947e110bc172ef7441fc680506d07ff"
INVALID_GEOMETRY = "ValueError: Nonfinite or out-of-image polygon vertex: no automatic clipping"
SCHEMA = "uv_three_region_test_mask_case_v1"
CATALOG_SCHEMA = "uv_three_region_test_catalog_v1"
TEST_COUNT = 21899
SIZE = 1024
TOLERANCE = 2e-6
DOMAINS = ("F","P","D","O")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(path, name, expected):
    path=Path(path).resolve()
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,"Frozen module changed: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def center_box(shape):
    h,w=shape
    require(type(h) is int and type(w) is int and min(h,w)>0,"Invalid native dimensions")
    side=min(h,w)
    left,top=(w-side)//2,(h-side)//2
    return [left,top,left+side,top+side]


def project_arrays(arrays, size=SIZE):
    """Original center-square geometry; float32 Pillow BOX, no binarization."""
    import numpy as np
    from PIL import Image
    require(set(arrays)==set(DOMAINS),"Require exactly F/P/D/O")
    shape=arrays["F"].shape
    require(len(shape)==2,"Native masks must be 2D")
    for value in arrays.values():
        require(value.shape==shape and value.dtype==np.bool_,"Masks must share native bool geometry")
    require(not np.any(arrays["P"] & arrays["D"]) and not np.any(arrays["P"] & arrays["O"]) and not np.any(arrays["D"] & arrays["O"]) and
            np.array_equal(arrays["P"] | arrays["D"] | arrays["O"],arrays["F"]),"Native partition overlaps or fails coverage")
    x0,y0,x1,y1=center_box(list(shape))
    weights,counts,cropped,outside,areas={}, {}, {}, {}, {}
    for name,value in arrays.items():
        crop=value[y0:y1,x0:x1]
        counts[name]=int(np.count_nonzero(value))
        cropped[name]=int(np.count_nonzero(crop))
        outside[name]=counts[name]-cropped[name]
        image=Image.fromarray(np.ascontiguousarray(crop,dtype=np.float32))
        projected=np.asarray(image.resize((size,size),resample=Image.Resampling.BOX),dtype=np.float32)
        require(projected.shape==(size,size) and np.isfinite(projected).all() and projected.min()>=0 and projected.max()<=1,"Invalid BOX coverage")
        weights[name]=np.ascontiguousarray(projected)
        areas[name]=float(np.sum(projected,dtype=np.float64))
    error=float(np.max(np.abs(weights["P"].astype(np.float64)+weights["D"]+weights["O"]-weights["F"])))
    require(error<=TOLERANCE,"Projected P+D+O does not equal F")
    side=x1-x0
    return weights,dict(native_shape=list(shape),projected_shape=[size,size],crop_box_xyxy=[x0,y0,x1,y1],
        counts_native=counts,counts_native_in_crop=cropped,counts_native_outside_crop=outside,
        effective_projected_area=areas,empty_projected_domains=[d for d in DOMAINS if areas[d]==0],
        projection_partition_max_abs_error=error,projection_partition_tolerance=TOLERANCE,
        projected_native_area_equivalent={d:areas[d]*(side/size)**2 for d in DOMAINS},
        projection_mass_relative_error={d:((areas[d]*(side/size)**2-cropped[d])/cropped[d] if cropped[d] else None) for d in DOMAINS},
        projection="center_square_floor_offsets_then_Pillow_float32_BOX",dtype="float32",
        additional_foreground_intersection=False,minimum_pixel_count_filter_applied=False)


def projection_worker(batch, request_path, output):
    import numpy as np
    import PIL
    from PIL import Image
    request=batch.read_json(request_path)
    require(request["projector_sha256"]==batch.sha(__file__),"Projection source changed")
    expected=request["runtime_versions"]
    require(str(np.__version__)==expected["numpy"] and str(PIL.__version__)==expected["PIL"],"Projection numpy/Pillow runtime differs")
    arrays={}
    for d in DOMAINS:
        path=batch.bound_file(Path(request_path).parent,request["native_masks"][d])
        with Image.open(path) as im:
            raw=np.asarray(im)
        require(raw.dtype==np.uint8 and raw.shape==tuple(request["native_shape"]) and np.all((raw==0)|(raw==255)),"Expected binary uint8 native mask")
        arrays[d]=raw!=0
    weights,audit=project_arrays(arrays)
    require(audit["counts_native"]==request["counts_native"],"Decoded native counts differ from frozen mask stage")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    target=output/"weights.npz"
    with target.open("xb") as stream:
        np.savez_compressed(stream,**weights)
    with np.load(target,allow_pickle=False) as loaded:
        require(set(loaded.files)==set(DOMAINS) and all(loaded[d].dtype==np.float32 and np.array_equal(loaded[d],weights[d]) for d in DOMAINS),"Compressed coverage roundtrip differs")
    audit.update(request_sha256=batch.sha(request_path),source_row_sha256=request["source_row_sha256"],
        contract_sha256=request["contract_sha256"],projector_sha256=request["projector_sha256"],
        runtime_versions=expected,native_mask_file_sha256={d:request["native_masks"][d]["sha256"] for d in DOMAINS},
        npz=batch.descriptor(target))
    batch.save_json(output/"projection.json",batch.seal(audit))


def validate_catalog(batch,builder,path,old):
    path=Path(path).resolve()
    catalog=batch.read_json(path)
    require(catalog.get("schema")==CATALOG_SCHEMA and catalog.get("status")=="frozen","Require frozen TEST catalog")
    source_path=batch.bound_file(path.parent,catalog["source_manifest"])
    require(catalog["source_manifest"]["sha256"]==batch.CLEAN_SHA,"TEST must use frozen clean manifest")
    source,_=builder.read_clean_manifest(source_path)
    test={r["sample_id"]:r for r in source if r["split"]=="test"}
    rows=catalog.get("rows",[])
    require(len(rows)==len(test)==TEST_COUNT and len({r["sample_id"] for r in rows})==TEST_COUNT,"Require every clean TEST group once; no sampling or replacement")
    root=Path(catalog["data_root"]).resolve()
    require(root==Path(old["cohort"]["data_root"]).resolve(),"Native source data_root differs from frozen runtime setup")
    normalized=[]
    for row in rows:
        original=test.get(row["sample_id"])
        require(original is not None and row.get("split")=="test" and row["subject_id"]==original["subject_id"] and row["native_shape"]==[int(original["source_height"]),int(original["source_width"])],"Catalog identity/native geometry differs from clean TEST")
        value=dict(row)
        for m in batch.MODS:
            desc=row[m]
            p=Path(desc["path"])
            p=(p if p.is_absolute() else root/p).resolve()
            expected=(root/original[m+"_path"]).resolve()
            require(p==expected and desc["sha256"]==original[m+"_sha256"] and batch.valid_sha(desc["sha256"]),"Catalog source path/hash differs from clean TEST")
            p.relative_to(root)
            value[m]=dict(path=str(p),sha256=desc["sha256"])
        normalized.append(value)
    return catalog,normalized


def copy_evidence(batch,native_case,case):
    """Archive source/stage JSON and logs. Native-image paths become audit-only."""
    destination=case/"evidence"
    hashes={}
    for source in sorted(native_case.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in (".json",".log"):
            continue
        require(not source.is_symlink(),"Symlink in private native evidence")
        rel=source.relative_to(native_case)
        target=destination/rel
        expected=batch.sha(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            require(batch.sha(target)==expected,"Archived evidence differs")
        else:
            with source.open("rb") as src,target.open("xb") as dst:
                shutil.copyfileobj(src,dst)
        require(batch.sha(target)==expected,"Evidence copy hash mismatch")
        hashes[str(rel.as_posix())]=dict(path=str(target.resolve()),sha256=expected)
    return hashes


def delete_owned_native(batch,root,index,owner,receipt_path):
    """The only destructive operation: an exact owned disposable case directory."""
    root=Path(root).resolve()
    parent=root/"private_native"/"cases"
    case=parent/("case_%06d"%index)
    if not case.exists():
        return
    require(not case.is_symlink() and not parent.is_symlink() and not (root/"private_native").is_symlink(),"Refuse symlinked temporary deletion target")
    resolved=case.resolve()
    require(resolved.parent==parent.resolve() and resolved.name=="case_%06d"%index and root in resolved.parents and len(resolved.relative_to(root).parts)==3,"Unsafe native temporary deletion target")
    require(batch.read_json(case/"owner.json")==owner,"Native directory ownership differs")
    receipt=batch.checked(batch.read_json(receipt_path))
    require(receipt.get("index")==index and receipt.get("contract_sha256")==owner["contract_sha256"] and receipt.get("sample_id")==owner["sample_id"] and receipt.get("status") in ("complete","no_unique_face","invalid_geometry"),"Do not remove native artifacts before sealed case publication")
    for desc in receipt["evidence"].values():
        batch.bound_file(root,desc)
    if receipt["status"]=="complete":
        batch.bound_file(root,receipt["npz"])
        batch.bound_file(root,receipt["projection_audit"])
    # 仅清理已核定路径
    shutil.rmtree(resolved)


def verify_compact(batch,settings,catalog,index,path):
    item=batch.checked(batch.read_json(path))
    row=settings["rows"][index]
    require(item.get("schema")==SCHEMA and item.get("index")==index and item.get("sample_id")==row["sample_id"] and item.get("subject_id")==row["subject_id"] and item.get("source_row_sha256")==batch.digest(catalog["rows"][index]) and item.get("contract_sha256")==batch.digest(settings["contract"]) and item.get("catalog_sha256")==settings["contract"]["catalog"]["sha256"],"Cached compact mask belongs to another source/protocol")
    require(item.get("sources")=={m:row[m] for m in batch.MODS} and item.get("native_shape")==row["native_shape"],"Cached native source descriptors differ")
    batch.verify_native(row)
    for desc in item["evidence"].values():
        batch.bound_file(path.parent,desc)
    if item["status"]=="complete":
        batch.bound_file(path.parent,item["npz"])
        audit=batch.checked(batch.read_json(batch.bound_file(path.parent,item["projection_audit"])))
        require(audit["npz"]==item["npz"] and audit["contract_sha256"]==item["contract_sha256"] and audit["source_row_sha256"]==item["source_row_sha256"] and audit["counts_native"]==item["counts_native"],"Cached projected coverage binding differs")
    else:
        require(item["status"] in ("no_unique_face","invalid_geometry") and item["npz"] is None and item["counts_native"] is None,"Unknown/fabricated technical result")
        log=batch.bound_file(path.parent,item["failure_stderr"])
        expected=batch.NO_FACE_ERROR if item["status"]=="no_unique_face" else INVALID_GEOMETRY
        require(batch.last_nonempty_line(log)==expected,"Technical failure evidence changed")
    return item


class StageJournal:
    """Do not expose old MI-specific eligible/excluded counters as TEST decisions."""
    def __init__(self,journal):
        self.journal=journal

    def emit(self,event,**fields):
        if event=="case_complete":
            fields=dict(fields)
            fields["original_mi_quota_flag_ignored"]=fields.pop("eligible",None)
            event="frozen_native_mask_case_generated"
        self.journal.emit(event,**fields)


def mark_published(journal,index,status):
    with journal.lock:
        journal.state["active"].pop(str(index),None)
        journal.state["completed"]=journal.state.get("completed",0)+1
        statuses=journal.state.setdefault("test_mask_status_counts",{})
        statuses[status]=statuses.get(status,0)+1


def process_one(batch,settings,catalog,index,journal,root):
    row=settings["rows"][index]
    case=root/"cases"/("case_%06d"%index)
    native=settings["output"]/"cases"/("case_%06d"%index)
    owner=dict(schema="uv_three_region_private_native_owner_v1",index=index,sample_id=row["sample_id"],contract_sha256=batch.digest(settings["contract"]))
    receipt_path=case/"receipt.json"
    if receipt_path.exists():
        item=verify_compact(batch,settings,catalog,index,receipt_path)
        delete_owned_native(batch,root,index,owner,receipt_path)
        mark_published(journal,index,item["status"])
        journal.emit("mask_case_cached",index=index,sample_id=row["sample_id"],mask_status=item["status"])
        return item
    native.mkdir(parents=True,exist_ok=True)
    batch.save_json(native/"owner.json",owner)
    case.mkdir(parents=True,exist_ok=True)
    status,failed_log="complete",None
    try:
        original=batch.process_case(settings,index,StageJournal(journal))
    except RuntimeError as exc:
        match=re.fullmatch(r"Stage failed, no sample excluded: "+re.escape(row["sample_id"])+r"/geometry exit=(-?\d+) log=(.+) stderr=(.+)",str(exc))
        if match is None:
            raise
        failed_log=Path(match.group(3)).resolve()
        require(failed_log.parent==native.resolve() and batch.last_nonempty_line(failed_log)==INVALID_GEOMETRY,"Unexpected geometry failure; fatal, not a technical exclusion")
        require(not (native/"case_receipt.json").exists() and not (native/"mask_receipt.json").exists(),"Invalid geometry has unexpected final mask receipt")
        batch.verify_native(row)
        original=None
        status="invalid_geometry"
    else:
        if original["status"]==batch.NO_FACE_STATUS:
            status="no_unique_face"
            stage=batch.checked(batch.read_json(native/"geometry_receipt.json"))
            batch.verify_technical_exclusion(stage)
            failed_log=Path(stage["stderr_path"]).resolve()
        else:
            require(original["status"]=="complete","Unknown original mask status")
            # 允许零像素区域
    base=dict(schema=SCHEMA,index=index,sample_id=row["sample_id"],subject_id=row["subject_id"],status=status,
        native_shape=row["native_shape"],source_row_sha256=batch.digest(catalog["rows"][index]),
        catalog_sha256=settings["contract"]["catalog"]["sha256"],contract_sha256=batch.digest(settings["contract"]),
        sources={m:row[m] for m in batch.MODS},artifact_kind="projected_region_coverage",
        minimum_pixel_count_filter_applied=False,native_artifacts_retention="private temporary; hash ledger archived; removed after sealed compact publication",
        metric_policy="Keep every TEST ID; empty region yields null per metric; technical failure yields unavailable; report valid N by metric")
    if status=="complete":
        request=dict(projector_sha256=batch.sha(__file__),native_masks=original["masks"],native_shape=row["native_shape"],
            counts_native=original["counts"],source_row_sha256=base["source_row_sha256"],contract_sha256=base["contract_sha256"],
            runtime_versions=settings["runtimes"]["mask"]["versions"])
        request_path=native/"projection_request.json"
        batch.save_json(request_path,request)
        project=case/"projection"
        if project.exists():
            # 保留未完成投影
            if not (project/"projection.json").exists():
                project.rename(case/("interrupted_projection_"+uuid.uuid4().hex))
            else:
                previous=batch.checked(batch.read_json(project/"projection.json"))
                require(previous.get("request_sha256")==batch.sha(request_path),"Existing projection belongs to different native masks")
                batch.bound_file(project,previous["npz"])
        if not project.exists():
            runtime=settings["runtimes"]["mask"]
            command=[runtime["python"],str(Path(__file__).resolve()),"--project-request",str(request_path),"--project-output",str(project),"--batch-module",str(settings["batch_module"])]
            with (native/("projection_"+uuid.uuid4().hex+".log")).open("xb") as stream:
                done=subprocess.run(command,env=batch.child_env(runtime),stdout=stream,stderr=subprocess.STDOUT,check=False)
            require(done.returncode==0,"Projection subprocess failed; native temporaries retained")
        projected=batch.checked(batch.read_json(project/"projection.json"))
        require(projected["request_sha256"]==batch.sha(request_path) and projected["counts_native"]==original["counts"],"Projection/native source binding differs")
        # 按描述记录定位NPZ
        base.update({key:projected[key] for key in ("npz","projected_shape","crop_box_xyxy","counts_native","counts_native_in_crop","counts_native_outside_crop","effective_projected_area","empty_projected_domains","projection_partition_max_abs_error","projection_partition_tolerance","dtype")})
        base.update(projection_audit=batch.descriptor(project/"projection.json"),mask_config_sha256=original["mask_config_sha256"],
            native_mask_file_sha256={d:original["masks"][d]["sha256"] for d in DOMAINS},failure_stderr=None)
    else:
        base.update(npz=None,projection_audit=None,counts_native=None,counts_native_in_crop=None,counts_native_outside_crop=None,
            effective_projected_area=None,empty_projected_domains=None,projected_shape=[SIZE,SIZE],crop_box_xyxy=center_box(row["native_shape"]),
            mask_config_sha256=None,native_mask_file_sha256=None)
    base["evidence"]=copy_evidence(batch,native,case)
    if failed_log is not None:
        base["failure_stderr"]=base["evidence"][str(failed_log.relative_to(native).as_posix())]
    item=batch.seal(base)
    batch.save_json(receipt_path,item)
    delete_owned_native(batch,root,index,owner,receipt_path)
    mark_published(journal,index,status)
    journal.emit("mask_case_published",index=index,sample_id=row["sample_id"],mask_status=status,
        empty_domains=item.get("empty_projected_domains"),receipt=str(receipt_path))
    return item


def run(args):
    batch_path=Path(args.batch_module).resolve()
    batch=load_module(batch_path,"frozen_test_mask_batch",CONTROLLER_SHA)
    builder=load_module(batch_path.with_name("mi_atlas_cohort_20261025.py"),"frozen_test_catalog_reader",BUILDER_SHA)
    old=batch.load_settings(args.frozen_config)
    require(batch.read_json(old["output"]/"batch_contract.json")==old["contract"],"Original source contract differs")
    catalog,rows=validate_catalog(batch,builder,args.catalog,old)
    root=Path(args.output_root).resolve()
    require(root!=old["output"] and old["output"] not in root.parents and root not in old["output"].parents,"TEST output must not contain or lie in original batch")
    for row in rows:
        require(all(root not in Path(row[m]["path"]).parents for m in batch.MODS),"Output contains original scientific input")
    require(1<=args.workers<=4,"Only1..4 CPU mask workers allowed")
    contract=dict(old["contract"])
    contract.update(schema="uv_three_region_test_mask_contract_v1",catalog=batch.descriptor(args.catalog),
        original_config=batch.descriptor(args.frozen_config),original_contract=batch.descriptor(old["output"]/"batch_contract.json"),
        mask_projector_sha256=batch.sha(__file__),source_row_order_sha256=batch.digest(catalog["rows"]),
        normalized_source_rows_sha256=batch.digest(rows),sample_order=[r["sample_id"] for r in rows],
        test_sample_count=len(rows),mask_workers=args.workers,projection=dict(size=SIZE,dtype="float32",resampling="Pillow_BOX_mode_F",crop="center_square_floor_offsets",partition_tolerance=TOLERANCE),
        eligibility="Every clean TEST row retained; no minimum region count; no replacement; metrics use own effective support and valid N",
        invalid_geometry_exception=INVALID_GEOMETRY,empty_domain_is_exclusion=False,minimum_pixel_count_filter_applied=False,
        source_row_digest_encoding="json.dumps ensure_ascii=False sort_keys=True indent=2 allow_nan=False plus newline UTF8")
    if sys.platform.startswith("linux") and os.getpriority(os.PRIO_PROCESS,0)<10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    with batch.batch_lock(root):
        batch.save_json(root/"mask_contract.json",contract)
        settings=dict(old,rows=rows,output=root/"private_native",contract=contract,workers=args.workers,batch_module=batch_path)
        journal=batch.Journal(dict(settings,output=root))
        results=[None]*len(rows)
        try:
            journal.emit("masks_running",test_samples=len(rows),minimum_pixel_count_filter_applied=False,workers=args.workers)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                active={}
                next_index=0
                while next_index<len(rows) and len(active)<args.workers:
                    active[pool.submit(process_one,batch,settings,catalog,next_index,journal,root)]=next_index
                    next_index+=1
                while active:
                    ready,_=wait(active,return_when=FIRST_COMPLETED)
                    for future in ready:
                        i=active.pop(future)
                        results[i]=future.result()
                    for _ in ready:
                        if next_index<len(rows):
                            active[pool.submit(process_one,batch,settings,catalog,next_index,journal,root)]=next_index
                            next_index+=1
            counts={status:sum(r["status"]==status for r in results) for status in ("complete","no_unique_face","invalid_geometry")}
            receipt=batch.seal(dict(schema="uv_three_region_test_mask_completion_v1",status="complete",catalog=batch.descriptor(args.catalog),
                contract=batch.descriptor(root/"mask_contract.json"),sample_count=len(rows),status_counts=counts,
                case_record_sha256={str(i):r["record_sha256"] for i,r in enumerate(results)},
                empty_domain_counts={d:sum(r["status"]=="complete" and d in r["empty_projected_domains"] for r in results) for d in DOMAINS},
                minimum_pixel_count_filter_applied=False,replacement_count=0,company_unchanged=True))
            batch.save_json(root/"receipt.json",receipt)
            journal.emit("complete",**counts,all_test_rows_accounted=True)
            return receipt
        except BaseException as exc:
            journal.emit("failed",error_type=type(exc).__name__,error=str(exc))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-module",type=Path,default=Path(__file__).with_name("mi_atlas_batch_20261025.py"))
    parser.add_argument("--catalog",type=Path)
    parser.add_argument("--frozen-config",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--project-request",type=Path)
    parser.add_argument("--project-output",type=Path)
    args=parser.parse_args()
    if args.project_request is not None:
        require(args.project_output is not None and args.catalog is None and args.frozen_config is None and args.output_root is None,"Projection subprocess arguments cannot mix controller mode")
        projection_worker(load_module(args.batch_module,"frozen_projection_io",CONTROLLER_SHA),args.project_request,args.project_output)
    else:
        require(args.catalog is not None and args.frozen_config is not None and args.output_root is not None and args.project_output is None,"Require --catalog --frozen-config --output-root")
        run(args)


if __name__=="__main__":
    main()
