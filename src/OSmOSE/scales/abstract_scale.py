import abc
import numpy as np
from dataclasses import dataclass

@dataclass
class AbstractScale(abc.ABC):
    sr : int = 312500

    @abc.abstractmethod
    def map_freq2scale(self, freq) -> float:
        return NotImplementedError

    @abc.abstractmethod
    def map_scale2freq(self, scale)-> float:
        return NotImplementedError
   
    def bbox2scale(self, t_start, t_end, f_min, f_max):
        f_min_projected_on_scale = self.map_freq2scale(f_min)
        f_max_projected_on_scale = self.map_freq2scale(f_max)
        return t_start, t_end, f_min_projected_on_scale, f_max_projected_on_scale

    def scale2bbox(self, t_start, t_end, y_min, y_max):
        f_min=self.map_scale2freq(y_min)
        f_max=self.map_scale2freq(y_max)
        return t_start, t_end, f_min, f_max
    
    def get_yticks(self):
        #TODO here insert a method to get the ticks value for display?.
        # Genre on pourrait décider que sur APLOSE on affiche toujours 12 ticks, 
        # et utiliser une instance de cet objet pour faire la convertion en frequence réelle si on est en log ou custom.
        return NotImplementedError