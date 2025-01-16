<template>
  <div >
    <el-container>
  <el-main style="background-color: #e9eef3;border-radius: 1.5vw;">
    <div style="display: flex;">    
      <div style="width: 70%;">
        <div style="height: 100%;width: 100%;">
          <div style="text-align: center;font-size: larger;font-weight: 900;color: black;background-color: #e9eef3;">可视化</div>
            <div v-if="isShowImg">            
              <img   :src="imgurl" alt="electron-vue" >
            </div>
            <div v-else>
              <img  style="height: 100%;width: 95%;" src="~@/assets/bg.png" alt="electron-vue" >
            </div>

        </div>      


      </div>
      <div style="width: 30%;height: 90vh;"><div style="height: 4vh;"></div><div style="height: 86vh;background-color: #ffffff;border-radius: 1.5vw;">
        <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">控制面板</div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-select v-model="value" placeholder="模式选择" @change="change">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>
          </el-select>
        </div>

        <div v-if="value==1">
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openFace" >
          <div style="width: 40%;text-align: center;">人脸检测</div>
          <el-switch
          v-model="isFace"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openPoint" >
            <div style="width: 40%;text-align: center;">关键点显示</div>
          <el-switch
          v-model="isPoint"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openAlign" >
            <div style="width: 40%;text-align: center;">人脸对齐</div>
          <el-switch
          v-model="isAlign"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="height: 20vh;"></div>
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
            <el-input style="width: 80%;" v-model="name" placeholder="请输入名字"></el-input>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="storageFace">录入人脸</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera">开启摄像头</el-button>
        </div>
        </div>
        <div v-if="value==2">
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openEye" >
          <div style="width: 40%;text-align: center;">眨眼检测</div>
          <el-switch
          v-model="isEye"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openMouth" >
          <div style="width: 40%;text-align: center;">张嘴检测</div>
          <el-switch
          v-model="isMouth"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openHead" >
          <div style="width: 40%;text-align: center;">摇头检测</div>
          <el-switch
          v-model="isHead"
          active-text="开启"
          inactive-text="关闭"
          >
          </el-switch>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  >
            <div style="width: 40%;text-align: center;">眨眼次数：</div>
            <div style="width: 32%;text-align: center;">{{ EyeCount }}</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  >
            <div style="width: 40%;text-align: center;">张嘴次数：</div>
            <div style="width: 32%;text-align: center;">{{ MouthCount }}</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" >
            <div style="width: 40%;text-align: center;">左摇头次数：</div>
            <div style="width: 32%;text-align: center;">{{ HeadLeftCount }}</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" >
            <div style="width: 40%;text-align: center;">右摇头次数：</div>
            <div style="width: 32%;text-align: center;">{{ HeadRightCount }}</div>
          </div>

          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  >
            <div style="width: 40%;text-align: center;">摇头次数：</div>
            <div style="width: 32%;text-align: center;">{{ HeadShakeCount }}</div>
          </div>


          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="reset_count">重置计数</el-button>
          </div>



        </div>
        <div v-if="value==3">
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openHand" >
            <div style="width: 40%;text-align: center;">手势识别</div>
            <el-switch
            v-model="isHand"
            active-text="开启"
            inactive-text="关闭"
            >
          </el-switch>
          </div>
            <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="openHandPoint" >
            <div style="width: 40%;text-align: center;">关键点显示</div>
            <el-switch
            v-model="isHandPoint"
            active-text="开启"
            inactive-text="关闭"
            >
          </el-switch>
          </div>
        </div>
      </div> </div>
    </div>
  </el-main>
</el-container>
  </div>
</template>

<script>
  import SystemInformation from './LandingPage/SystemInformation'

  export default {
    name: 'landing-page',
    components: { SystemInformation },
    data () {
      return {
        //
        EyeCount : 0,
        MouthCount : 0,
        HeadLeftCount : 0,
        HeadRightCount : 0,
        HeadShakeCount : 0,
        isShowImg:false,
        isFace:false,
        isPoint:false,
        isAlign:false,
        isEye:false,
        isMouth:false,
        isHead:false,
        isGettingCount:false,
        isHand:false,
        isHandPoint:false,
        name:'',
        imgurl_:'http://localhost:8000/video',
        timestamp: Date.now(),
        options: [{
          value: 1,
          label: '人脸操作'
        }, {
          value: 2,
          label: '活体检测'
        }, {
          value: 3,
          label: '手势识别'
        }],
        value: 1
      }
    },
    mounted () {
      //
    },
    computed: {
      //
      imgurl(){
        return `${this.imgurl_}?t=${this.timestamp}`;
      }
    },
    watch: {
      //
    },
    methods: {

      turnOnGetCount() {
        this.isGettingCount = true;
        this.intervalId = setInterval(() => {
          this.getCount();
          if (!this.isGettingCount) {
            clearInterval(this.intervalId);
          }
        }, 200);
      },
      turnoffGetCount() {
        this.isGettingCount = false;
      },  
      controlGetCount(sign){
        if(sign){
          console.log(this.isEye,this.isMouth,this.isHead);
          if((this.isHead && !this.isMouth && !this.isEye) ||
    (!this.isHead && this.isMouth && !this.isEye) ||
    (!this.isHead && !this.isMouth && this.isEye))
          {
            
            this.turnOnGetCount();
          }
        }else{
          if(!this.isHead && !this.isMouth && !this.isEye)
          {
            this.turnoffGetCount();
            console.log("turnoffGetCount");
          }
        }
        
      },

      getCount()
      {
        this.$http.get('http://127.0.0.1:8000/get_count')
        .then(response => {
          console.log(response.data);
          this.EyeCount = response.data.EyeCount;
          this.MouthCount = response.data.MouthCount;
          this.HeadLeftCount = response.data.HeadLeftCount;
          this.HeadRightCount = response.data.HeadRightCount;
          this.HeadShakeCount = response.data.HeadShakeCount;
        })
      },
      change(){
        let data = {'weight':this.value }
        this.$http.post('http://127.0.0.1:8000/weight',data)
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      },
      open (link) {
        this.$electron.shell.openExternal(link)
      },

      openPoint(){
        this.$http.get('http://127.0.0.1:8000/turn_point')
        .then(response => {
          if (response.data.status == 1)
            {
                this.$data.isPoint = true;
            }
          else{
                this.$data.isPoint = false;
          }
          console.log(response.data,this.isPoint);
        })
      },
      openAlign(){
        this.$http.get('http://127.0.0.1:8000/turn_align')
        .then(response => {
          if (response.data.status == 1)
            {
                this.$data.isAlign = true;
            }
          else{
                this.$data.isAlign = false;
          }
          console.log(response.data,this.isAlign);
        })
      },
      openFace(){
        this.$http.get('http://127.0.0.1:8000/turn_face')
        .then(response => {
          if (response.data.status == 1)
            {
                this.$data.isFace = true;
            }
          else{
                this.$data.isFace = false;
          }
          console.log(response.data,this.isFace);
        })
        .catch(error => {
         console.error(error);
        })
      },
      openEye(){
        this.$http.get('http://127.0.0.1:8000/turn_eye')
        .then(response => {
          if (response.data.status == 1)
            {
                this.controlGetCount(1)
                this.$data.isEye = true;
            }
          else{
                this.$data.isEye = false;
                this.controlGetCount(0)
          }
          console.log(response.data,this.isEye);
        })
      },
      openMouth(){
        this.$http.get('http://127.0.0.1:8000/turn_mouth')
        .then(response => {
          if (response.data.status == 1)
            {
                this.controlGetCount(1)
                this.$data.isMouth = true;
            }
          else{
                this.$data.isMouth = false;
                this.controlGetCount(0)
          }
          console.log(response.data,this.isMouth);
        })
      },
      openHead(){
        this.$http.get('http://127.0.0.1:8000/turn_head')
        .then(response => {
          if (response.data.status == 1)
            {
              this.controlGetCount(1)
                this.$data.isHead = true;
            }
          else{
                this.$data.isHead = false;
                this.controlGetCount(0)
          }
          console.log(response.data,this.isHead);
        })
      },
      openHand(){
        this.$http.get('http://127.0.0.1:8000/turn_hand')
        .then(response => {
          if (response.data.status == 1)
            {
                this.$data.isHand = true;
            }
          else{
                this.$data.isHand = false;
          }
          console.log(response.data,this.isHand);
        })
      },
      openHandPoint(){
        this.$http.get('http://127.0.0.1:8000/turn_hand_point')
        .then(response => {
          if (response.data.status == 1)
            {
                this.$data.isHandPoint = true;
            }
          else{
                this.$data.isHandPoint = false;
          }
          console.log(response.data,this.isHandPoint);
        })
      },
      reset_count(){
        this.$http.get('http://127.0.0.1:8000/reset_count')
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      },
      openCamera(){
        this.$http.get('http://127.0.0.1:8000/turn_camera')
        .then(response => {
          if (response.data.status == 1)
            {
                this.isShowImg = true;
            }
          else{
                this.isShowImg = false;
          }
          console.log(response.data.status,this.isShowImg);
        })
        .catch(error => {
          console.error(error);
        });
      },
      storageFace(){
        this.$message({
          message: '请保证摄像头前只有一个人',
          type: 'warning'
        });
        if (!this.isAlign){
          this.openAlign()
        }
        let data = {'name':this.name}
        this.$http.post('http://127.0.0.1:8000/storage_face',data)
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      }
    }
  }
</script>

<style>
  @import url('https://fonts.googleapis.com/css?family=Source+Sans+Pro');



  body { font-family: "Helvetica Neue",Helvetica,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","微软雅黑",Arial,sans-serif;}






</style>
