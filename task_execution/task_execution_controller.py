from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from vision.vision_controller import VisionController 
from protocol.vnc_controller import VNCController
from emulator.emulator_controller import EmulatorController
from task_execution.event_manager import EventManager
from task_execution.task_planner import TaskPlanner
from task_execution.data_structures import TaskContext, StepPlan
from task_execution.step_planner import StepPlanner
from models.openrouter import OpenRouter
from vision.vision_analysis import ScreenAnalysis
import json
import time
from threading import Thread
from task_execution.data_structures import PlanVersion
from models.api import Model

class TaskExecutionController:
    """
    Controls the execution of automated tasks by coordinating between
    vision analysis, task planning, and event execution.
    """
    
    def __init__(self, 
                 vision_controller: VisionController,
                 vnc_controller: VNCController,
                 emulator_controller: EmulatorController,
                 debug_mode: bool = False):
        
        
        self.planning_model = Model.create_from_config("planning_model")
        self.vision = vision_controller
        self.vnc = vnc_controller
        self.emulator = emulator_controller
        self.event_manager = EventManager(vnc_controller)
        self.task_planner = TaskPlanner()
        self.step_panner = StepPlanner()
        
        self.max_retries = 3
        self.retry_delay = 2.0
        self.current_context: Optional[TaskContext] = None
        self.frame_stability_timeout = 60.0  # seconds
        self.screenshot_thread = None
        self.debug_mode = debug_mode


    def execute_task(self, task_description: str) -> bool:
        """Main entry point for task execution"""
        # try:
        self.emulator.run()
        self.vnc.connect()

        self.screenshot_thread = Thread(target=self.vision.process_continuous_screenshots)
        self.screenshot_thread.daemon = True  # Thread will terminate when main program exits
        self.screenshot_thread.start()

        print("For inital stable frame...")
        screen_analysis = self.vision.wait_for_next_stable_frame(self.frame_stability_timeout)
        if screen_analysis is None:
            raise Exception("Timeout waiting for stable frame")
    
        initial_plan = self.task_planner.create_execution_plan(task_description, screen_analysis)
        initial_plan.print_plan()
        print(f"Initial plan is_complete: {initial_plan.is_complete}")

        # Create initial plan version
        initial_version = PlanVersion(
            plan=initial_plan,
            created_at=time.time(),
            reason="Initial plan creation",
            step_mapping=None  # No previous plan to map from
        )

        # Initialize TaskContext with plan versions
        self.current_context = TaskContext(
            user_task=task_description,
            plan_versions=[initial_version],  # Start with initial plan
            current_step=0,
            current_phase="initializing",
            observations=[],
            step_histories={}  # Empty dict to store execution histories
        )

        previous_step_plan = None
        previous_step_plan_success = False
        
        while True:

            if screen_analysis is None:
                raise Exception("No valid frame available")

            verdict = self.task_planner.evaluate_progress(
                self.current_context, 
                screen_analysis, 
                previous_step_plan, 
                previous_step_plan_success
            )

            if verdict:
                print("Task completed successfully")
                print(self.current_context.create_execution_summary())
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
            
            # Take snapshot and execute action
            # snapshot_name = f"step_{len(self.current_context.attempted_actions)}"
            success, screen_analysis = self._try_execute_action(screen_analysis, plan)

            previous_step_plan = plan
            previous_step_plan_success = success
                # self.current_context.attempted_actions.append(next_actions)
                # print(f"Taking snapshot: {snapshot_name}")

        # except Exception as e:
        #     if self.screenshot_thread and self.screenshot_thread.is_alive():
        #         self.vision.stop_processing = True
        #         self.screenshot_thread.join(timeout=1.0)
            
        #     emulator_controller.stop()
        #     print(f"Task execution failed: {str(e)}")
        #     return False  
    
    def _try_execute_action(self, current_screen: ScreenAnalysis, plan: StepPlan) -> Tuple[bool, Optional[ScreenAnalysis]]:
        """Try to execute an action with retries"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                if self.debug_mode:
                    print("\nPlanned actions:")
                    plan.print_plan()
                    if not self._show_pre_execution_menu(current_screen, plan):
                        return False, None

                success = self.event_manager.execute_instructions(plan)
                if not success:
                    return False, None
                
                time.sleep(0.5)
                self.vision.discard_stable_frame()
                
                next_screen = self.vision.wait_for_next_stable_frame(self.frame_stability_timeout)
                if next_screen is None:
                    raise Exception("Timeout waiting for stable frame")
                
                if self.debug_mode:
                    print("\nAction executed. Analyzing result...")
                    self._show_post_execution_menu(next_screen)
                
                return success, next_screen
                    
            except Exception as e:
                print(f"Action execution failed: {str(e)}")
            
            retry_count += 1
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                
        return False, None

    def get_execution_context(self) -> Optional[TaskContext]:
        """Get current execution context"""
        return self.current_context
    
    def _show_debug_menu(self, screen_analysis: ScreenAnalysis) -> None:
        """Interactive debug menu for execution control"""
        while True:
            print("\n=== Debug Menu ===")
            print("1. Continue execution")
            print("2. Create QEMU snapshot")
            print("3. Save visual debug output")
            print("4. Show current screen analysis")
            print("q. Quit execution")
            
            choice = input("\nEnter choice (1-4, q): ").strip().lower()
            
            if choice == '':
                break
            elif choice == '2':
                snapshot_name = input("Enter snapshot name: ").strip()
                if snapshot_name:
                    self.emulator.create_snapshot(snapshot_name)
                    print(f"Created snapshot: {snapshot_name}")
            elif choice == '3':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                debug_dir = Path("debug_output")
                debug_dir.mkdir(exist_ok=True)
                
                # Save screenshot
                screenshot_path = debug_dir / f"screen_{timestamp}.png"
                self.vision.save_current_screenshot(screenshot_path)
                
                # Save analysis data
                analysis_path = debug_dir / f"analysis_{timestamp}.json"
                with open(analysis_path, 'w') as f:
                    json.dump(screen_analysis.to_dict(), f, indent=2)
                    
                print(f"Saved debug output to {debug_dir}")
            elif choice == '4':
                print("\nCurrent Screen Analysis:")
                print(f"Text elements found: {len(screen_analysis.text_screen_info)}")
                print(f"UI elements found: {len(screen_analysis.ui_elements)}")
                print("\nText content:")
                for elem in screen_analysis.text_screen_info:
                    print(f"- {elem.text} at {elem.bounds}")
            elif choice == 'q':
                raise KeyboardInterrupt("Debug execution stopped by user")

    def _show_pre_execution_menu(self, current_screen: ScreenAnalysis, plan: StepPlan) -> bool:
        """Interactive debug menu before action execution"""
        while True:
            print("\n=== Pre-Execution Debug Menu ===")
            print("1. Execute planned actions")
            print("2. Create QEMU snapshot")
            print("3. Show detailed plan")
            print("4. Show execution context")
            print("5. Save visual debug data")
            print("q. Quit execution")
            
            choice = input("\nEnter choice (1-5, q): ").strip().lower()
            
            if choice == '':
                return True
            elif choice == '2':
                snapshot_name = input("Enter snapshot name: ").strip()
                if snapshot_name:
                    self.emulator.create_snapshot(snapshot_name)
                    print(f"Created snapshot: {snapshot_name}")
            elif choice == '3':
                print("\nDetailed Plan:")
                plan.print_plan()
            elif choice == '4':
                if self.current_context:
                    print("\nCurrent Execution Context:")
                    print(self.current_context.create_execution_summary())
                else:
                    print("No execution context available")
            elif choice == '5':
                try:
                    # Get current screen analysis
                    debug_path = current_screen.save_debug_info(plan)
                    print(f"\nSaved debug data to: {debug_path}")
                except Exception as e:
                    print(f"Failed to save debug data: {str(e)}")
            elif choice == 'q':
                raise KeyboardInterrupt("Debug execution stopped by user")

    def _show_post_execution_menu(self, screen_analysis: ScreenAnalysis) -> None:
        """Interactive debug menu after action execution"""
        while True:
            print("\n=== Post-Execution Debug Menu ===")
            print("1. Continue to next action")
            print("2. Create QEMU snapshot")
            print("3. Save visual debug data")
            print("4. Show current screen analysis")
            print("5. Show execution context")
            print("q. Quit execution")
            
            choice = input("\nEnter choice (1-5, q): ").strip().lower()
            
            if choice == '':
                break
            elif choice == '2':
                snapshot_name = input("Enter snapshot name: ").strip()
                if snapshot_name:
                    self.emulator.create_snapshot(snapshot_name)
                    print(f"Created snapshot: {snapshot_name}")
            elif choice == '3':
                try:
                    debug_path = screen_analysis.save_debug_info(None)
                    print(f"\nSaved debug data to: {debug_path}")
                except Exception as e:
                    print(f"Failed to save debug data: {str(e)}")
            elif choice == '4':
                print("\nCurrent Screen Analysis:")
                print(f"Text elements found: {len(screen_analysis.text_elements)}")
                print(f"UI elements found: {len(screen_analysis.ui_elements)}")
                print("\nText content:")
                for elem in screen_analysis.text_elements:
                    print(f"- {elem.text} at {elem.bounds}")
            elif choice == '5':
                if self.current_context:
                    print("\nCurrent Execution Context:")
                    print(self.current_context.create_execution_summary())
                else:
                    print("No execution context available")
            elif choice == 'q':
                raise KeyboardInterrupt("Debug execution stopped by user")

if __name__ == "__main__":
    # set -gx PYTHONPATH $PYTHONPATH /Users/klaus/Library/CloudStorage/OneDrive-Persönlich/Coding/Emulation/emulator-control-ng
    from vision.vision_google import GoogleVision
    from vision.vision_qwen import QwenVision
    import json

    from emulator.emulator_controller import EmulatorController
    from emulator.computer import QemuComputer, Drive, DriveType, DriveInterface
    import time
    install_iso = Path("data/en_windows_xp_sp3.iso")
    cdrom_path = Path("data/WINDOWS_XP_1.iso")
    disk_path = Path("data/new_disk.qcow2")

    computer = QemuComputer()
    computer.add_drive(Drive(
        path=install_iso,
        drive_type=DriveType.CDROM,
        readonly=True,
        boot=True
    ))

    computer.add_drive(Drive(
        path=Path("data/new_disk.qcow2"), 
        drive_type=DriveType.HDD,
        interface=DriveInterface.IDE
    ))

    emulator_controller = EmulatorController(computer)
    print("computer ", computer.to_qemu_args())
    # emulator_controller.run()
    # emulator_controller.restore_snapshot("test")
    # time.sleep(100)

 
    vnc_controller = VNCController()

    vision_controller = VisionController(vnc_controller=vnc_controller)
     
    task_execution_controller = TaskExecutionController(
        vision_controller=vision_controller, 
        vnc_controller=vnc_controller, 
        emulator_controller=emulator_controller,
        debug_mode=True)
    
    task_description = """
       Install Windows XP. the license key is MRX3F-47B9T-2487J-KWKMF-RPWBY. if other input valeus are needed, choose appropriate values for a test environment.
 """

    #task_description = """
    #    The task is to install Microsoft Works 8.5. The programm is located on the CDROM drive. The CDROM drive is usually drive D. 
    #    The installation files are located in the APPS folder. 
    #"""

    success = task_execution_controller.execute_task(task_description)
    if success:
        print("Task execution completed successfully")
    else:
        print("Task execution failed")