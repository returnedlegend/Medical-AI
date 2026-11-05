#!/usr/bin/env python3
"""Real-browser integration checks for the stateful synthetic procedure laboratory.

Start the application first. This creates retained research episodes and one small
real training run; it never substitutes fabricated injury or learning responses.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chromium", default=shutil.which("chromium"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/retraction-procedure-browser")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    errors = []
    expect.set_options(timeout=20000)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=args.chromium,
            headless=True,
            args=[
                "--enable-webgl",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1050}, accept_downloads=True
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.base_url, wait_until="networkidle")

        def assert_rendered(label):
            viewport = page.locator(".three-tissue-viewport canvas, .tissue-scene")
            expect(viewport).to_be_visible()
            variation = viewport.evaluate(
                """async element => {
                let pixels,width,height;
                if(element instanceof HTMLCanvasElement){
                    const gl=element.getContext('webgl2');
                    if(!gl || gl.isContextLost()) return 0;
                    width=gl.drawingBufferWidth;height=gl.drawingBufferHeight;
                    pixels=new Uint8Array(width*height*4);
                    gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
                } else {
                    const image=new Image();
                    image.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new XMLSerializer().serializeToString(element));
                    await image.decode();
                    const canvas=document.createElement('canvas');
                    width=canvas.width=800;height=canvas.height=600;
                    const context=canvas.getContext('2d');context.drawImage(image,0,0,width,height);
                    pixels=context.getImageData(0,0,width,height).data;
                }
                const sums=[0,0,0],squared=[0,0,0];let count=0;
                for(let y=Math.floor(height*.15);y<height*.9;y+=3)for(let x=Math.floor(width*.1);x<width*.9;x+=3){
                    const offset=(y*width+x)*4;count++;
                    for(let c=0;c<3;c++){sums[c]+=pixels[offset+c];squared[c]+=pixels[offset+c]**2;}
                }
                return sums.reduce((total,sum,c)=>total+Math.sqrt(Math.max(0,squared[c]/count-(sum/count)**2)),0);
            }"""
            )
            assert (
                variation > 40
            ), f"{label}: anatomy viewport was blank or insufficiently rendered (RGB variation {variation:.2f})"
            viewport.screenshot(path=str(args.output_dir / f"{label}-viewport.png"))

        expect(
            page.get_by_role("button", name="New episode", exact=True)
        ).to_be_visible()

        def click_response(button, path):
            expect(button).to_be_enabled()
            with page.expect_response(
                lambda r: r.request.method == "POST"
                and r.url.split("?")[0].endswith(path)
            ) as captured:
                button.click()
            response = captured.value
            assert response.ok, f"{path}: {response.status} {response.text()}"
            return response.json()

        creation_writes = []

        def record_creation_write(request):
            if request.method in ("POST", "DELETE"):
                creation_writes.append((request.method, urlparse(request.url).path))

        page.on("request", record_creation_write)
        page.get_by_label("Procedure random seed", exact=True).fill("2")
        dev_initial = click_response(
            page.get_by_role("button", name="New episode", exact=True), "/sessions"
        )
        assert dev_initial["seed"] == 2
        expect(
            page.get_by_role("log", name="Procedure event timeline", exact=True)
        ).to_contain_text(dev_initial["id"][:8])
        expect(page.locator(".mini-trace .trace-range").first).to_contain_text(
            "1 samples"
        )
        initial_events = page.locator(".event-log > div").count()
        initial_trace_labels = page.locator(
            ".mini-trace .trace-range"
        ).all_text_contents()
        failure = {"pending": True}

        def fail_replacement_once(route):
            if route.request.method == "POST" and failure["pending"]:
                failure["pending"] = False
                route.fulfill(
                    status=503,
                    content_type="application/json",
                    body=json.dumps({"detail": "Injected replacement episode outage"}),
                )
            else:
                route.continue_()

        context.route("**/api/procedure/sessions", fail_replacement_once)
        with page.expect_response(
            lambda response: response.request.method == "DELETE"
        ) as closed:
            with page.expect_response(
                lambda response: response.request.method == "POST"
                and urlparse(response.url).path == "/api/procedure/sessions"
            ) as failed_replacement:
                page.get_by_role("button", name="New episode", exact=True).click()
        assert closed.value.ok and closed.value.json()["status"] == "closed"
        assert failed_replacement.value.status == 503
        expect(page.get_by_role("alert")).to_contain_text(
            "Injected replacement episode outage"
        )
        expect(page.locator(".viewport-caption")).to_contain_text("closed")
        expect(
            page.get_by_label("Saved episode", exact=True).locator("option:checked")
        ).to_contain_text("closed")
        for label in ("Move instrument right", "Step", "Handoff"):
            expect(page.get_by_role("button", name=label, exact=True)).to_be_disabled()
        assert page.locator(".event-log > div").count() == initial_events
        assert (
            page.locator(".mini-trace .trace-range").all_text_contents()
            == initial_trace_labels
        )
        persisted_closed = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{dev_initial['id']}"
        ).json()
        assert persisted_closed["status"] == "closed"
        assert not context.request.get(
            f"{args.base_url}/api/procedure/sessions/{dev_initial['id']}/transitions"
        ).json()["transitions"]
        writes_before_retry = len(creation_writes)
        page.wait_for_timeout(500)
        assert len(creation_writes) == writes_before_retry
        page.screenshot(
            path=str(args.output_dir / "closed-after-replacement-failure.png"),
            full_page=True,
        )
        retried = click_response(
            page.get_by_role("button", name="New episode", exact=True), "/sessions"
        )
        assert retried["seed"] == 2 and retried["status"] == "active"
        assert sum(method == "DELETE" for method, _ in creation_writes) == 1
        context.unroute("**/api/procedure/sessions", fail_replacement_once)
        page.remove_listener("request", record_creation_write)
        checks.append(
            "close-before-create failure preserves truthful read-only state and history; explicit retry creates once without duplicate events"
        )
        # Resume the suite with its established development seed2.
        initial = click_response(
            page.get_by_role("button", name="New episode", exact=True), "/sessions"
        )
        episode = initial["id"]
        assert len(initial["state"]["tissue"]["nodes"]) == 23
        page.wait_for_selector(".three-tissue-viewport canvas, .tissue-scene")
        page.wait_for_timeout(600)
        assert_rendered("initial")
        page.screenshot(
            path=str(args.output_dir / "procedure-initial-desktop.png"), full_page=True
        )
        checks.append(
            "actual episode initialization and state-driven 3D or XY rendering"
        )
        reading_mutations = []

        def record_reading_mutation(request):
            if request.method == "POST" and request.url.endswith("/step"):
                reading_mutations.append(request.url)

        page.on("request", record_reading_mutation)
        event_log = page.get_by_role("log", name="Procedure event timeline", exact=True)
        event_log.focus()
        event_log.press("ArrowDown")
        event_log.press("Space")
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Space")
        page.wait_for_timeout(200)
        after_reading = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{episode}"
        ).json()
        assert (
            not reading_mutations and after_reading["revision"] == initial["revision"]
        )
        assert after_reading["state"]["tools"] == initial["state"]["tools"]
        page.remove_listener("request", record_reading_mutation)
        checks.append(
            "reading keys outside the focused viewport do not move instruments or mutate the episode"
        )
        closed = click_response(
            page.get_by_role("button", name=re.compile("^Close grasper")), "/step"
        )
        assert closed["state"]["tools"]["grasper"]["contact_node_id"] is not None
        page.get_by_label("Instrument motion magnitude", exact=True).focus()
        page.get_by_label("Instrument motion magnitude", exact=True).press("End")
        moved = closed
        for _ in range(4):
            moved = click_response(
                page.get_by_role("button", name="Move instrument right", exact=True),
                "/step",
            )
        assert moved["state"]["tools"]["grasper"]["force_n"] > 0
        assert (
            moved["state"]["tissue"]["nodes"][3]["position_mm"]
            != initial["state"]["tissue"]["nodes"][3]["position_mm"]
        )
        assert moved["reward_components"]
        checks.append(
            "manual grasp and pull update mechanics, node geometry and reward components"
        )
        assert_rendered("retraction")
        page.screenshot(
            path=str(args.output_dir / "procedure-retraction-desktop.png"),
            full_page=True,
        )
        reset = click_response(
            page.get_by_role("button", name="Reset", exact=True), "/reset"
        )
        assert reset["id"] != episode and reset["revision"] == 0
        checks.append("reset creates a fresh episode while retaining old history")
        page.locator(".tool-select button").nth(1).click()
        for _ in range(12):
            heated = click_response(
                page.get_by_role("button", name="Move instrument right", exact=True),
                "/step",
            )
        for _ in range(4):
            heated = click_response(
                page.get_by_role("button", name="Move instrument up", exact=True),
                "/step",
            )
        page.get_by_label("Cautery power", exact=True).focus()
        page.get_by_label("Cautery power", exact=True).press("End")
        for _ in range(12):
            heated = click_response(
                page.get_by_role("button", name=re.compile("^Cautery pulse")), "/step"
            )
            if heated["terminated"] or heated["truncated"]:
                break
        assert max(n["damage"] for n in heated["state"]["tissue"]["nodes"]) > 0
        assert heated["state"]["physiology"]["blood_loss_ml"] > 0
        assert heated["state"]["physiology"]["heart_rate_bpm"] > 72
        assert heated["state"]["physiology"]["mean_arterial_pressure_mmhg"] < 90
        assert heated["episode_reward"] < 0
        fetched = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{heated['id']}"
        ).json()
        assert fetched["state"]["tissue"]["nodes"] == heated["state"]["tissue"]["nodes"]
        assert_rendered("injury")
        injury_view = page.locator(".three-tissue-viewport canvas, .tissue-scene")
        frozen_injury = injury_view.screenshot()
        page.wait_for_timeout(700)
        assert (
            injury_view.screenshot() == frozen_injury
        ), "Terminal injury indicators advanced while the model clock was stopped"
        checks.append(
            "terminal anatomy and blood markers remain frozen with the stored model clock"
        )
        page.screenshot(
            path=str(args.output_dir / "procedure-injury-desktop.png"), full_page=True
        )
        checks.append(
            "contact heating causes persistent real engine injury, bleeding, physiological response and negative reward"
        )
        with page.expect_download() as downloaded:
            page.get_by_role("link", name="Episode JSON", exact=True).click()
        download = downloaded.value
        download.save_as(args.output_dir / "injury-episode.json")
        exported = json.loads((args.output_dir / "injury-episode.json").read_text())
        assert exported
        checks.append("episode export is downloadable and contains retained data")
        page.get_by_label("Saved episode", exact=True).select_option(episode)
        expect(
            page.get_by_role("log", name="Procedure event timeline")
        ).to_contain_text("Restored")
        page.get_by_label("Saved episode", exact=True).select_option(heated["id"])
        expect(
            page.get_by_role("log", name="Procedure event timeline")
        ).to_contain_text("Restored")
        page.get_by_role("button", name="Reward trace", exact=True).click()
        expect(
            page.get_by_role("img", name=re.compile("Observed reward per transition"))
        ).to_be_visible()
        checks.append(
            "saved episodes restore recorded transitions, events and reward traces"
        )
        assert_rendered("restored-injury")
        recording = page.get_by_label("Recorded transition", exact=True)
        recording.focus()
        recording.press("Home")
        expect(page.locator(".replay-state")).to_have_text("Read-only recorded frame")
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_disabled()
        assert_rendered("recorded-early-frame")
        retained_after_inspection = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{heated['id']}"
        ).json()
        assert retained_after_inspection["revision"] == heated["revision"]
        assert (
            retained_after_inspection["state"]["tissue"]["nodes"]
            == heated["state"]["tissue"]["nodes"]
        )
        page.get_by_role("button", name="Return to latest", exact=True).click()
        expect(page.locator(".replay-state")).to_have_text("Latest saved state")
        checks.append(
            "recorded-state inspection updates actual anatomy without changing the saved episode"
        )
        page.get_by_role("button", name="Events", exact=True).click()
        reset = click_response(
            page.get_by_role("button", name="Reset", exact=True), "/reset"
        )
        page.keyboard.press("g")
        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.endswith("/step")
        ) as keyboard_step:
            page.keyboard.press("Space")
        assert keyboard_step.value.ok
        keyboard_state = keyboard_step.value.json()
        assert keyboard_state["state"]["tools"]["grasper"]["grip"] > 0
        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.endswith("/step")
        ) as keyboard_motion:
            page.keyboard.press("ArrowRight")
        assert keyboard_motion.value.ok
        handoff = click_response(
            page.get_by_role("button", name="Handoff", exact=True), "/step"
        )
        assert (
            handoff["terminated"]
            and handoff["state"]["termination_reason"] == "handoff"
        )
        checks.append(
            "keyboard selection, grasp and motion work; handoff records an actual terminal decision"
        )
        terminal_deletes = []

        def record_terminal_delete(request):
            if request.method == "DELETE":
                terminal_deletes.append(request.url)

        page.on("request", record_terminal_delete)
        click_response(
            page.get_by_role("button", name="New episode", exact=True), "/sessions"
        )
        page.remove_listener("request", record_terminal_delete)
        assert not terminal_deletes
        terminal_record = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{handoff['id']}"
        ).json()
        assert terminal_record["status"] == "terminated"
        expect(
            page.get_by_label("Saved episode", exact=True).locator(
                f'option[value="{handoff["id"]}"]'
            )
        ).to_contain_text("terminated")
        checks.append(
            "switching from a terminal episode preserves its displayed and durable terminal status"
        )
        reset = click_response(
            page.get_by_role("button", name="Reset", exact=True), "/reset"
        )
        page.get_by_label("Playback controller", exact=True).select_option("baseline")
        run = page.get_by_role("button", name="Run policy", exact=True)
        expect(run).to_be_enabled()
        run.click()
        page.wait_for_timeout(1500)
        page.get_by_label("Procedure random seed", exact=True).focus()
        page.keyboard.press("Escape")
        expect(
            page.get_by_role("button", name="Run policy", exact=True)
        ).to_be_visible()
        page.wait_for_timeout(500)
        paused = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{reset['id']}"
        ).json()
        assert paused["revision"] > 0
        page.wait_for_timeout(450)
        stable = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{reset['id']}"
        ).json()
        assert stable["revision"] == paused["revision"]
        checks.append(
            "scripted playback advances actual actions and Escape from a form input stops subsequent steps"
        )
        page.get_by_role("button", name="Policy learning", exact=True).click()
        page.get_by_label("Learning algorithm", exact=True).select_option("dqn")
        page.get_by_label("Training episodes", exact=True).fill("20")
        trained = click_response(
            page.get_by_role("button", name="Train policy", exact=True),
            "/learning/runs",
        )
        training_id = trained["id"]
        expect(page.locator(".training-status")).to_contain_text(
            "completed", timeout=180000
        )
        checkpoint = context.request.get(
            f"{args.base_url}/api/learning/runs/{training_id}/results"
        ).json()
        expect(
            page.get_by_role("region", name="Held-out policy evaluation").get_by_role(
                "row"
            )
        ).to_have_count(5)
        expect(
            page.get_by_role("img", name=re.compile("Actual training returns over"))
        ).to_be_visible()
        learned_row = (
            page.get_by_role("region", name="Held-out policy evaluation")
            .get_by_role("row")
            .filter(
                has=page.get_by_role("rowheader", name="Learned policy", exact=True)
            )
        )
        expect(learned_row).to_contain_text(
            f"{checkpoint['evaluation']['learned']['success_rate']*100:.1f}%"
        )
        expect(learned_row).to_contain_text(
            f"{checkpoint['evaluation']['learned']['severe_harm_rate']*100:.1f}%"
        )
        checks.append(
            "held-out policy evaluation and training curve show actual completed rollouts"
        )
        recovery_page = context.new_page()
        recovery_page.on("pageerror", lambda e: errors.append(str(e)))
        recovery_writes = []
        recovery_page.on(
            "request",
            lambda request: (
                recovery_writes.append(request.url)
                if request.method not in ("GET", "HEAD", "OPTIONS")
                else None
            ),
        )
        recovery_attempts = {"spec": 0, "results": 0}

        def transient_failure(kind):
            def handle(route):
                recovery_attempts[kind] += 1
                if recovery_attempts[kind] == 1:
                    route.fulfill(
                        status=503,
                        json={"detail": f"Transient {kind} service failure"},
                    )
                else:
                    route.continue_()

            return handle

        recovery_page.route("**/procedure/spec", transient_failure("spec"))
        recovery_page.route(
            f"**/learning/runs/{training_id}/results", transient_failure("results")
        )
        recovery_page.goto(args.base_url, wait_until="networkidle")
        recovery_page.get_by_role("button", name="Policy learning", exact=True).click()
        recovery_page.get_by_label("Policy checkpoint", exact=True).select_option(
            training_id
        )
        expect(
            recovery_page.get_by_role(
                "button", name="Retry engine specification", exact=True
            )
        ).to_be_visible()
        expect(
            recovery_page.get_by_role(
                "button", name="Retry checkpoint results", exact=True
            )
        ).to_be_visible()
        expect(
            recovery_page.get_by_text("Archived model checkpoint", exact=True)
        ).to_have_count(0)
        recovery_page.screenshot(
            path=str(args.output_dir / "transient-checkpoint-unavailable.png"),
            full_page=True,
        )
        recovery_page.get_by_role(
            "button", name="Retry checkpoint results", exact=True
        ).click()
        expect(
            recovery_page.get_by_role(
                "button", name="Retry checkpoint results", exact=True
            )
        ).to_have_count(0)
        expect(
            recovery_page.get_by_role("button", name="New matched episode", exact=True)
        ).to_be_disabled()
        expect(
            recovery_page.get_by_text("Archived model checkpoint", exact=True)
        ).to_have_count(0)
        checks.append(
            "a failed checkpoint-results GET recovers through explicit retry without duplicate work"
        )
        recovery_page.get_by_role(
            "button", name="Retry engine specification", exact=True
        ).click()
        expect(
            recovery_page.get_by_role("button", name="New matched episode", exact=True)
        ).to_be_enabled()
        assert recovery_attempts == {"spec": 2, "results": 2}
        assert not recovery_writes, recovery_writes
        recovery_page.screenshot(
            path=str(args.output_dir / "transient-checkpoint-recovered.png"),
            full_page=True,
        )
        recovery_page.close()
        checks.append(
            "an unavailable engine specification is distinguished from an archived checkpoint and retries recover eligibility"
        )
        matched = click_response(
            page.get_by_role("button", name="New matched episode", exact=True),
            f"/learning/runs/{training_id}/session",
        )
        assert not matched["policy_context"]["out_of_distribution"]
        assert matched["config"] == checkpoint["checkpoint"]["environment_config"]
        expect(page.locator(".policy-context")).to_contain_text(
            "Environment parameters match checkpoint"
        )
        checks.append(
            "matched episodes use the actual checkpoint environment and report matching parameters"
        )
        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.endswith("/policy-step")
        ) as policy_step:
            page.get_by_role("button", name="Run policy", exact=True).click()
        assert policy_step.value.ok
        assert not policy_step.value.json()["policy_context"]["out_of_distribution"]
        pause = page.get_by_role("button", name="Pause", exact=True)
        if pause.count():
            pause.click()
        page.wait_for_timeout(350)
        policy_state = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{matched['id']}"
        ).json()
        assert policy_state["revision"] > 0
        checks.append(
            "actual DQN training completes and checkpoint playback advances the matched environment"
        )
        page.screenshot(
            path=str(args.output_dir / "procedure-policy-desktop.png"), full_page=True
        )
        race_episode = click_response(
            page.get_by_role("button", name="New matched episode", exact=True),
            f"/learning/runs/{training_id}/session",
        )
        reference = context.request.get(
            f"{args.base_url}/api/learning/reference"
        ).json()
        assert reference["available"] and reference["run_id"] != training_id
        held_policy = []
        race_route = f"**/sessions/{race_episode['id']}/policy-step"

        def hold_actual_policy_response(route):
            held_policy.append((route, route.fetch()))

        page.route(race_route, hold_actual_policy_response)
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
        page.get_by_role("button", name="Step", exact=True).click()
        for _ in range(200):
            if held_policy:
                break
            page.wait_for_timeout(50)
        assert len(held_policy) == 1
        with page.expect_response(
            lambda r: r.request.method == "GET"
            and "/policy-context?" in r.url
            and reference["run_id"] in r.url
        ) as switched_context:
            page.get_by_label("Policy checkpoint", exact=True).select_option(
                reference["run_id"]
            )
        assert switched_context.value.ok
        assert switched_context.value.json()["out_of_distribution"]
        expect(page.locator(".policy-context")).to_contain_text("max_steps")
        completed_action = held_policy[0][1].json()
        held_policy[0][0].fulfill(response=held_policy[0][1])
        page.unroute(race_route, hold_actual_policy_response)
        expect(page.locator(".viewport-caption")).to_contain_text(
            f"revision {completed_action['revision']}"
        )
        expect(page.get_by_label("Policy checkpoint", exact=True)).to_have_value(
            reference["run_id"]
        )
        expect(page.locator(".policy-context")).to_contain_text(
            "Outside training conditions"
        )
        expect(page.locator(".policy-context")).to_contain_text("max_steps")
        expect(page.locator(".policy-context-error")).to_have_count(0)
        page.screenshot(
            path=str(args.output_dir / "late-policy-response-context.png"),
            full_page=True,
        )
        checks.append(
            "late policy responses preserve the real body transition without overwriting a newly selected checkpoint context"
        )
        outside = context.request.post(
            f"{args.base_url}/api/procedure/sessions",
            data={"seed": 74, "config": {"max_steps": 20}},
        )
        assert outside.ok
        outside_episode = outside.json()
        context_gate = {"mode": "hold", "requests": []}
        context_route = f"**/sessions/{outside_episode['id']}/policy-context?*"

        def gate_policy_context(route):
            if context_gate["mode"] == "hold":
                context_gate["requests"].append(route)
            elif context_gate["mode"] == "fail":
                route.abort("connectionrefused")
            else:
                route.continue_()

        page.route(context_route, gate_policy_context)
        page.reload(wait_until="networkidle")
        page.get_by_label("Saved episode", exact=True).select_option(
            outside_episode["id"]
        )
        page.get_by_role("button", name="Policy learning", exact=True).click()
        page.get_by_label("Policy checkpoint", exact=True).select_option(training_id)
        page.get_by_label("Playback controller", exact=True).select_option("trained")
        page.wait_for_timeout(150)
        assert context_gate["requests"]
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_disabled()
        expect(
            page.get_by_role("button", name="Run policy", exact=True)
        ).to_be_disabled()
        unchanged = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{outside_episode['id']}"
        ).json()
        assert unchanged["revision"] == 0
        context_gate["mode"] = "real"
        for held in context_gate["requests"]:
            held.continue_()
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
        checks.append(
            "single-step and automatic policy inference wait for actual environment comparison"
        )
        context_gate["mode"] = "fail"
        page.reload(wait_until="networkidle")
        page.get_by_label("Saved episode", exact=True).select_option(
            outside_episode["id"]
        )
        page.get_by_role("button", name="Policy learning", exact=True).click()
        page.get_by_label("Policy checkpoint", exact=True).select_option(training_id)
        page.get_by_label("Playback controller", exact=True).select_option("trained")
        expect(page.locator(".policy-context-error")).to_contain_text(
            "could not be checked"
        )
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_disabled()
        expect(
            page.get_by_role("button", name="Run policy", exact=True)
        ).to_be_disabled()
        context_gate["mode"] = "real"
        page.get_by_role("button", name="Retry policy context", exact=True).click()
        expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
        expect(page.locator(".policy-context-error")).to_have_count(0)
        page.unroute(context_route, gate_policy_context)
        checks.append(
            "failed policy context blocks inference until explicit retry recovers"
        )
        expect(page.locator(".policy-context-notice")).to_contain_text(
            "outside training conditions"
        )
        expect(page.locator(".policy-context")).to_contain_text("max_steps")
        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.endswith("/policy-step")
        ) as outside_step:
            page.get_by_role("button", name="Run policy", exact=True).click()
        assert (
            outside_step.value.ok
            and outside_step.value.json()["policy_context"]["out_of_distribution"]
        )
        pause = page.get_by_role("button", name="Pause", exact=True)
        if pause.count():
            pause.click()
        page.wait_for_timeout(350)
        retained = context.request.get(
            f"{args.base_url}/api/procedure/sessions/{outside_episode['id']}"
        ).json()
        assert retained["policy_context"]["out_of_distribution"]
        checks.append(
            "changed environment conditions are warned before playback and retained with policy provenance"
        )
        page.get_by_role("button", name="Tissue / tools", exact=True).click()
        canvas = page.locator(".three-tissue-viewport canvas")
        if canvas.count():
            canvas.evaluate(
                "canvas => canvas.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext()"
            )
            expect(page.locator(".tissue-scene")).to_be_visible()
            assert_rendered("webgl-loss-fallback")
            page.get_by_role("button", name="3D surface", exact=True).click()
            expect(
                page.get_by_role("button", name="Reset 3D camera", exact=True)
            ).to_be_visible()
            assert_rendered("restored-3d-renderer")
            checks.append(
                "WebGL loss falls back to rendered XY state and the 3D camera can be restored"
            )
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(args.output_dir / "procedure-mobile.png"), full_page=True
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        checks.append("mobile layout has no horizontal document overflow")
        page.locator(".desktop-tabbar").get_by_role(
            "button", name="Model documentation", exact=True
        ).click()
        expect(
            page.get_by_role("heading", name="6. Live environment defaults and bounds")
        ).to_be_visible()
        expect(
            page.get_by_role("region", name="Live environment parameters")
        ).to_contain_text("force limit n")
        checks.append(
            "in-app methods expose stateful response equations and live engine parameter bounds"
        )
        assert not errors, errors
        browser.close()
    report = {
        "checks": checks,
        "javascript_errors": errors,
        "training_run_id": training_id,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
