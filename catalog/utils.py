from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import re
import time
import datetime

class folder_utils:
    def __init__(self, logger, config):
        self.img_path=  config.get('general', 'img_path')
        self.logger = logger
        self.months = config.get('general', 'months').split(',')
        self.cat_ar = config.get('general', 'cat_ar').split(',')
        self.cat_ni = config.get('general', 'cat_ni').split(',')
        self.img_path_out = config.get('general', 'img_path_out')
        self.sens = config.get('general', 'path_sens')
        self.config = config

    def catalog(self, img_path, people, prev_category, prev_ts):
        m,y,d,timestamp = self.extract_date(img_path)
        category = self.extract_category(people, prev_category)
        sens= self.sens in img_path
        if category != None:
            self.logger.info('Extracted cat is ' + category)
            if prev_category != None and category != prev_category and timestamp - prev_ts < 86400 :
                self.move_file(img_path, prev_category, y, m)
            else :
                self.move_file(img_path, category, y, m)
        elif prev_category != None and (not sens):
            self.logger.info('No extracted category')
            if timestamp - prev_ts < 86400 :
                self.logger.info('Photo in the same day as before, cat: '+prev_category)
                self.move_file(img_path, prev_category, y, m)
                category = prev_category
            else:
                self.logger.info('Move to date path')
                self.move_file(img_path, '', y, m)
        else:
            self.logger.info('No extracted category')
            self.logger.info('Move to date path')
            self.move_file(img_path, '', y, m)

        return category, timestamp

    def catalog_woface(self, img_path, prev_category, prev_ts):
        m,y,d,timestamp = self.extract_date(img_path)
        self.logger.info('ts:'+ str(timestamp) + ', prev ts:' + str(prev_ts))
        if (timestamp - prev_ts < 86400) and prev_category != None:
            self.logger.info('Time photo is in the same day as before, category: ' + str(prev_category) + ' ts:'+ str(timestamp) + ', prev ts:' + str(prev_ts))
            self.move_file(img_path, prev_category, y, m)
        else :
            self.logger.info('Move to date path')
            self.move_file(img_path, '', y, m)

        return timestamp

    def move_file(self, img_path, category, y, m):
        new_path=os.path.join(self.img_path_out, category, str(y), str(self.months[m-1]))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.move(img_path, os.path.join(new_path, os.path.basename(img_path)))
        self.logger.info('Moved file '+img_path+ ' to '+new_path)

    def validate(self, date_text, format):
        try:
            datetime.datetime.strptime(date_text, format)
            return True
        except ValueError:
            try:
                datetime.datetime.strptime(date_text, format.replace('-',''))
                return True
            except:
                return False

    def extract_date(self, img_path):
        hour = minute = second = 0
        date = re.search("([0-9]{2}\-?[0-9]{2}\-?[0-9]{4})", img_path)
        date2 = re.search("([0-9]{4}\-?[0-9]{2}\-?[0-9]{2})", img_path)
        if date != None and self.validate(date.groups()[0], '%d-%m-%Y')==True:
            self.logger.info('Date1:'+str(date.groups()[0]))
            if '-' in date.groups()[0] :
                date_arr = date.groups()[0].split('-')
                year = date_arr[2]
                month = date_arr[1]
                day = date_arr[0]
            else:
                month= date.groups()[0][4:6]
                year= date.groups()[0][0:4]
                day = date.groups()[0][6:8]
        elif date2 != None and self.validate(date2.groups()[0], '%Y-%m-%d')==True:
            self.logger.info('Date2:'+str(date2.groups()[0]))
            if '-' in date2.groups()[0] :
                date_arr = date2.groups()[0].split('-')
                year = date_arr[0]
                month = date_arr[1]
                day = date_arr[2]
            else:
                month= date2.groups()[0][4:6]
                year= date2.groups()[0][0:4]
                day= date2.groups()[0][6:8]
        else:
            created = os.path.getctime(img_path)
            year,month,day,hour,minute,second=time.localtime(created)[:-3]

        self.logger.info('year:'+str(year)+', month:'+str(month))

        timest = None
        hourmin = re.search("[|_]([0-2][0-9]{5})", img_path)
        if hourmin != None:
            hour= hourmin.groups()[0][0:2]
            minute= hourmin.groups()[0][2:4]
            second= hourmin.groups()[0][4:6]
            try:
                dt = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
            except:
                dt = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=0, minute=0, second=0)

            timest= time.mktime(dt.timetuple())
            self.logger.info('Time:'+str(hourmin.groups()[0]))
        else:
            if hour >0 and minute > 0 and second > 0:
                dt = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
                timest= time.mktime(dt.timetuple())
            else:
                created = os.path.getctime(img_path)
                y,m,d,h,m,s=time.localtime(created)[:-3]
                dt = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(h), minute=int(m), second=int(s))
                timest= time.mktime(dt.timetuple())
                self.logger.info('Time:'+str(timest))


        self.logger.info('Timestamp:'+str(timest))

        return int(month),int(year),int(day),int(timest)

    def extract_category(self, people, prev_category):
        clause1 = clause2 = clause3 = clause4 = clause5 = clause6 = clause7 = 0
        for person in people:
            if eval(self.config.get('general', 'clause1')):
                clause1+=1
            if eval(self.config.get('general', 'clause7')):
                clause7+=1
            if eval(self.config.get('general', 'clause2')):
                clause2+=1
            if eval(self.config.get('general', 'clause3')):
                clause3+=1
            if eval(self.config.get('general', 'clause4')):
                clause4+=1
            if eval(self.config.get('general', 'clause5')):
                clause5+=1
            if eval(self.config.get('general', 'clause6')):
                clause6+=1

        if clause1 > 0 :
            return self.config.get('general', 'response1')
        if clause7 > 0 :
            return self.config.get('general', 'response7')
        if clause2 > 0 :
            return self.config.get('general', 'response2')
        if clause3 > 0 :
            return self.config.get('general', 'response3')
        if clause4 > 0 :
            return self.config.get('general', 'response4')
        if clause5 > 0 :
            return self.config.get('general', 'response5')
        if clause6 > 0 :
            return self.config.get('general', 'response6')

        return None
