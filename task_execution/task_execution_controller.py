from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from vision.vision_controller import VisionController 
from protocol.vnc_controller import VNCController
from emulator.emulator_controller import EmulatorController
from task_execution.event_manager import EventManager
from task_execution.task_planner import TaskPlanner
from task_execution.data_structures import TaskContext, StepPlan
from task_execution.step_planner import StepPlanner
from task_execution.step_evaluator import StepEvaluator
from models.openrouter import OpenRouter
from vision.vision_analysis import ScreenAnalysis
import json
import time
from threading import Thread

class TaskExecutionController:
    """
    Controls the execution of automated tasks by coordinating between
    vision analysis, task planning, and event execution.
    """
    
    def __init__(self, 
                 model: OpenRouter,
                 vision_controller: VisionController,
                 vnc_controller: VNCController,
                 emulator_controller: EmulatorController):
        
        
        self.vision = vision_controller
        self.vnc = vnc_controller
        self.emulator = emulator_controller
        self.event_manager = EventManager(model, vnc_controller)
        self.task_planner = TaskPlanner(model)
        self.step_panner = StepPlanner(model)
        self.step_evaluator = StepEvaluator(model)
        
        self.max_retries = 3
        self.retry_delay = 2.0
        self.current_context: Optional[TaskContext] = None
        self.frame_stability_timeout = 60.0  # seconds
        self.screenshot_thread = None


    def execute_task(self, task_description: str) -> bool:
        """Main entry point for task execution"""
        try:
            self.emulator.run()
            self.vnc.connect()

            self.screenshot_thread = Thread(target=self.vision.process_continuous_screenshots)
            self.screenshot_thread.daemon = True  # Thread will terminate when main program exits
            self.screenshot_thread.start()

            print("For inital stable frame...")
            screen_analysis = self.vision.wait_for_next_stable_frame(self.frame_stability_timeout)
            if screen_analysis is None:
                raise Exception("Timeout waiting for stable frame")
        
            plan = self.task_planner.create_execution_plan(task_description, screen_analysis)
            plan.print_plan()
            print(plan.is_complete)

            self.current_context = TaskContext(
                plan=plan,
                user_task=task_description,
                current_step = 0,
                current_phase="initializing", 
                observations=[],
                attempted_actions=[]
            )

            while True:
                
                verdict = self.task_planner.evaluate_progress(self.current_context, screen_analysis)
                if verdict:
                    print("Task completed successfully")
                    return True
                    
                print("Let's plan the next actions...")
                plan = self.step_panner.determine_next_action(
                    self.current_context,
                    screen_analysis
                )

                if plan is None:
                    print("No plan found")
                    continue

                next_actions = plan.steps           
                if not next_actions:
                    print("No actions found")
                    continue
                
                     
                print(f"plan: ", plan.plan_description)
                print(f"Plan outcome: {plan.expected_outcome}")
                
                for action in next_actions:
                    print(f"Action: {action.action}")
                    print(f"Context: {action.context}")
                    print()
                    
                # Take snapshot and execute action
                snapshot_name = f"step_{len(self.current_context.attempted_actions)}"
                success, screen_analysis = self._try_execute_action(plan, screen_analysis)

                if not success:
                    return False

                self.current_context.attempted_actions.append(next_actions)
                # print(f"Taking snapshot: {snapshot_name}")

        except Exception as e:
            if self.screenshot_thread and self.screenshot_thread.is_alive():
                self.vision.stop_processing = True
                self.screenshot_thread.join(timeout=1.0)
            
            emulator_controller.stop()
            print(f"Task execution failed: {str(e)}")
            return False  
    
    def _try_execute_action(self, plan: StepPlan, current_screen: ScreenAnalysis) -> bool:
        """Try to execute an action with retries"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                success = True 
                success = self.event_manager.execute_instructions(plan.steps)
                if not success:
                    return False
                
                time.sleep(0.5)
                vision_controller.discard_stable_frame()
                
                next_screen = vision_controller.wait_for_next_stable_frame(self.frame_stability_timeout)
                if next_screen is None:
                    raise Exception("Timeout waiting for stable frame")
                
                return success, next_screen
                    
            except Exception as e:
                print(f"Action execution failed: {str(e)}")
            
            retry_count += 1
            if retry_count < self.max_retries:
                # self.emulator.restore_snapshot(snapshot_name)
                time.sleep(self.retry_delay)
                
        return False

    def get_execution_context(self) -> Optional[TaskContext]:
        """Get current execution context"""
        return self.current_context
    
if __name__ == "__main__":
    from vision.vision_google import GoogleVision
    from vision.vision_qwen import QwenVision
    import json

    from emulator.emulator_controller import EmulatorController
    from emulator.computer import QemuComputer, Drive, DriveType, DriveInterface
    import time
    cdrom_path = Path("data/en_windows_xp_sp3.iso")
    disk_path = Path("data/new_disk.qcow2")

    computer = QemuComputer()
 #   computer.add_drive(Drive(
 #       path=Path("data/en_windows_xp_sp3.iso"),
 #       drive_type=DriveType.CDROM,
 #       readonly=True,
 #       boot=True
 #   ))

    computer.add_drive(Drive(
        path=Path("data/xp_disk.qcow2"), 
        drive_type=DriveType.HDD,
        interface=DriveInterface.IDE
    ))

    emulator_controller = EmulatorController(computer)
    print("computer ", computer.to_qemu_args())
    # emulator_controller.run()
    # emulator_controller.restore_snapshot("test")
    # time.sleep(100)winalso nwei

    with open('config.json') as config_file:
        config = json.load(config_file)

    openrouter_api_key = config.get("openrouter_api_key")
    openrouter_model = OpenRouter(openrouter_api_key, "google/gemini-2.0-flash-001")

    api_key = config.get("google_api_key")
    openrouter_api_key = config.get("openrouter_api_key")
    alibaba_api_key = config.get("alibaba_api_key")

    vision_elements = GoogleVision(api_key)
    vision_coords = QwenVision(alibaba_api_key)
    vnc_controller = VNCController()

    vision_controller = VisionController(
        vnc_controller=vnc_controller,
        model=openrouter_model,
        vision_provider_elements=vision_elements,
        vision_provider_coords=vision_coords)
   
    task_execution_controller = TaskExecutionController(
        model=openrouter_model,
        vision_controller=vision_controller, 
        vnc_controller=vnc_controller, 
        emulator_controller=emulator_controller)
    
    # task_description = """
    #   Install Windows XP. the license key is MRX3F-47B9T-2487J-KWKMF-RPWBY. if other input valeus are needed, choose appropriate values for a test environment.
    # """

    task_description = """
        make sure that the screensaver is disabled.
    """

    success = task_execution_controller.execute_task(task_description)
    if success:
        print("Task execution completed successfully")
    else:
        print("Task execution failed")